#ifndef LPM_FTLE_HPP
#define LPM_FTLE_HPP

#include "LpmConfig.h"
#include "lpm_geometry.hpp"
#include "lpm_logger.hpp"
#include "mesh/lpm_mesh_seed.hpp"
#include "util/lpm_floating_point.hpp"
#include "util/lpm_math.hpp"

namespace Lpm {

template <typename SeedType>
struct ComputeFTLE {
  using face_kind = typename SeedType::faceKind;
  using geo = typename SeedType::geo;
  using crd_view = typename geo::crd_view_type;
  using face_vertex_view =  Kokkos::View<Index * [face_kind::nverts]>;

  static_assert(std::is_same<face_kind, QuadFace>::value,
    "FTLE for non-quadrilateral faces not implemented yet.");

  static constexpr Real fp_tol = 1e-14;

  scalar_view_type ftle; /// output
  crd_view phys_crds_verts; /// input
  crd_view ref_crds_verts; /// input
  crd_view phys_crds_faces; /// input
  crd_view ref_crds_faces; /// input
  face_vertex_view face_verts; /// input
  mask_view_type face_mask; /// input
  Real t; // input

  ComputeFTLE(scalar_view_type ftle,
    const crd_view phys_crds_verts,
    const crd_view ref_crds_verts,
    const crd_view phys_crds_faces,
    const crd_view ref_crds_faces,
    const face_vertex_view face_verts,
    const mask_view_type face_mask,
    const Real& time_since_ref ) :
    ftle(ftle),
    phys_crds_verts(phys_crds_verts),
    ref_crds_verts(ref_crds_verts),
    phys_crds_faces(phys_crds_faces),
    ref_crds_faces(ref_crds_faces),
    face_verts(face_verts),
    face_mask(face_mask),
    t(time_since_ref) {}

  template <typename Tensor2Type, typename EdgeVecType, typename UnitVecType>
  KOKKOS_INLINE_FUNCTION
  void set_flow_map_gradient(
    Tensor2Type& mat,
    const EdgeVecType& e0reverse_phys,
    const EdgeVecType& e1phys,
    const UnitVecType& xdir,
    const UnitVecType& ydir,
    const Real& dx0,
    const Real& dy0) const {
    mat[0] = PlaneGeometry::dot(e1phys, xdir) / dx0;
    mat[1] = PlaneGeometry::dot(e0reverse_phys, xdir) / dx0;
    mat[2] = PlaneGeometry::dot(e1phys, ydir) / dy0;
    mat[3] = PlaneGeometry::dot(e0reverse_phys, ydir) / dy0;
  }

  template <typename Tensor2Type>
  KOKKOS_INLINE_FUNCTION
  void cauchy_green_tensor(Tensor2Type& cg_tensor, const Tensor2Type& flow_map_gradient) const {
    for (int i=0; i<2; ++i) {
      for (int j=0; j<2; ++j) {
        const int ij_idx = 2*i + j;
        const int ji_idx = 2*j + i;
        cg_tensor[ij_idx] = flow_map_gradient[ij_idx] * flow_map_gradient[ji_idx];
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index face_idx) const {
    return impl<typename SeedType::geo>(face_idx);
  }

  template <typename Geo>
  KOKKOS_INLINE_FUNCTION typename std::enable_if<
  std::is_same<Geo, SphereGeometry>::value, void>::type
  impl(const Index face_idx) const {
    /*
    *   Spherical FTLE
    */
    if (!face_mask(face_idx)) { // skip faces that have been divided

      // get coordinates of face center in reference space and physical space
      const auto fai = Kokkos::subview(ref_crds_faces, face_idx, Kokkos::ALL);
      const auto fxi = Kokkos::subview(phys_crds_faces, face_idx, Kokkos::ALL);
      // normalize physical space which may contain time discretization error
      // (reference space will always lie exactly on the sphere)
      SphereGeometry::normalize(fxi);

      // build the rotation matrix for both face vectors
      Kokkos::Tuple<Real,9> rot_mat_ref = north_pole_rotation_matrix(fai);
      Kokkos::Tuple<Real,9> rot_mat_phys = north_pole_rotation_matrix(fxi);
#ifndef NDEBUG
      Real npole_phys[3];
      Real npole_ref[3];

      apply_3by3(npole_ref, rot_mat_ref, fai);
      apply_3by3(npole_phys, rot_mat_phys, fxi);

      for (int i=0; i<3; ++i) {
        LPM_KERNEL_ASSERT( FloatingPoint<Real>::zero(npole_phys[0], fp_tol) );
        LPM_KERNEL_ASSERT( FloatingPoint<Real>::zero(npole_phys[1], fp_tol) );
        LPM_KERNEL_ASSERT( FloatingPoint<Real>::equiv(npole_phys[2], 1, fp_tol) );
        LPM_KERNEL_ASSERT( FloatingPoint<Real>::zero(npole_ref[0], fp_tol) );
        LPM_KERNEL_ASSERT( FloatingPoint<Real>::zero(npole_ref[1], fp_tol) );
        LPM_KERNEL_ASSERT( FloatingPoint<Real>::equiv(npole_ref[2], 1, fp_tol) );
      }
#endif
      // collect the face's vertex coordinates
      Real vert_phys[face_kind::nverts][geo::ndim];
      Real vert_ref[face_kind::nverts][geo::ndim];
      Index vert_idxs[face_kind::nverts];
      for (int i=0; i<face_kind::nverts; ++i) {
        const Index vert_idx = face_verts(face_idx, i);
        vert_idxs[i] = vert_idx;
        for (int j=0; j<geo::ndim; ++j) {
          vert_phys[i][j] = phys_crds_verts(vert_idx, j);
          vert_ref[i][j] = ref_crds_verts(vert_idx, j);
        }
      }

      // apply the rotation matrices
      Real face_phys_tangent_crds[3] = {0,0,1};
      Real face_ref_tangent_crds[3] = {0,0,1};
      Real vert_phys_tangent_crds[face_kind::nverts][geo::ndim];
      Real vert_ref_tangent_crds[face_kind::nverts][geo::ndim];
      for (int i=0; i<face_kind::nverts; ++i) {
        apply_3by3(vert_ref_tangent_crds[i], rot_mat_ref, vert_ref[i]);
        apply_3by3(vert_phys_tangent_crds[i], rot_mat_phys, vert_phys[i]);
      }

      // shift (arbitrary choice)
      // so that vertex 1's coordinates define the origin
      for (int i=0; i<2; ++i) {
        face_phys_tangent_crds[i] -= vert_phys_tangent_crds[1][i];
        face_ref_tangent_crds[i] -= vert_ref_tangent_crds[1][i];
      }
      for (int i=0; i<4; ++i) {
        for (int j=0; j<2; ++j) {
          vert_phys_tangent_crds[i][j] -= vert_phys_tangent_crds[1][j];
          vert_ref_tangent_crds[i][j] -= vert_ref_tangent_crds[1][j];
        }
      }
      const Real edge1_ref[2] = {
         vert_ref_tangent_crds[2][0] - vert_ref_tangent_crds[1][0],
         vert_ref_tangent_crds[2][1] - vert_ref_tangent_crds[1][1]};
      const Real edge1_phys[2] = {
         vert_phys_tangent_crds[2][0] - vert_phys_tangent_crds[1][0],
         vert_phys_tangent_crds[2][1] - vert_phys_tangent_crds[1][1]};
      const Real dx0 = PlaneGeometry::mag(edge1_ref);
      // define the local "x" direction by reference edge 1, which points from
      // vertex 1 to vertex 2
      Real xdir_ref[2];
      // normalize
      Real xref_r=0;
      for (int i=0; i<2; ++i) {
        xdir_ref[i] = edge1_ref[i];
        xref_r += square(xdir_ref[i]);
      }
      xref_r = sqrt(xref_r);
      for (int i=0; i<2; ++i) {
        xdir_ref[i] /= xref_r;
      }

      // local "y" is then points approximately to vertex 0
      // along the reverse direction of reference edge 0;
      // it needs to be orthogonalized
      const Real edge0_rev_ref[2] = {vert_ref_tangent_crds[0][0], vert_ref_tangent_crds[0][1]};
      const Real edge0_rev_phys[2]= {vert_phys_tangent_crds[0][0], vert_phys_tangent_crds[0][1]};
      Real ydir_ref[2];
      // orthogonalize
      Real dot_xy_ref = 0;
      for (int i=0; i<2; ++i) {
        dot_xy_ref += xdir_ref[i]*edge0_rev_ref[i];
      }
      for (int i=0; i<2; ++i) {
        ydir_ref[i] = edge0_rev_ref[i] - dot_xy_ref * xdir_ref[i];
      }
      const Real dy0 = PlaneGeometry::mag(ydir_ref);
      // normalize
      Real yref_r = 0;
      for (int i=0; i<2; ++i) {
        yref_r += square(ydir_ref[i]);
      }
      yref_r = sqrt(yref_r);
      for (int i=0; i<2; ++i) {
        ydir_ref[i] /= yref_r;
      }
      LPM_KERNEL_ASSERT(
        FloatingPoint<Real>::zero(PlaneGeometry::dot(xdir_ref, ydir_ref), fp_tol) );

      Real flow_map_gradient[4];
      set_flow_map_gradient(flow_map_gradient, edge0_rev_phys, edge1_phys, xdir_ref, ydir_ref, dx0, dy0);
      Real cg_tensor[4];
      cauchy_green_tensor(cg_tensor, flow_map_gradient);

      const auto eigs = two_by_two_real_eigenvalues(cg_tensor);
      const Real lambda1 = eigs[0];
      const Real lambda2 = eigs[1];
#ifndef NDEBUG
//       if (!FloatingPoint<Real>::equiv(lambda1*lambda2, 1, fp_tol)) {
//         spdlog::warn("ftle error: abs(lambda1 * lambda2 - 1)= {} (should be 0)",
//           abs(lambda1*lambda2-1));
//       }
#endif
//         LPM_KERNEL_ASSERT(FloatingPoint<Real>::equiv(lambda1*lambda2, 1, fp_tol));

      // TODO: divide by time here?
      ftle(face_idx) = log(lambda1);// * FloatingPoint<Real>::safe_denominator(2*t);
    }
  }

    template <typename Geo>
    KOKKOS_INLINE_FUNCTION typename std::enable_if<
    std::is_same<Geo, PlaneGeometry>::value, void>::type
    impl(const Index face_idx) const {
      /*
      * Planar FTLE
      */
      if (!face_mask(face_idx)) { // skip faces that have been divided

        // collect face coordinates
        Real face_ref[geo::ndim];
        Real face_phys[geo::ndim];
        for (int i=0; i<2; ++i) {
          face_ref[i] = ref_crds_faces(face_idx, i);
          face_phys[i] = phys_crds_faces(face_idx, i);
        }

        // collect the face's vertex indices and coordinates
        Real vert_phys[face_kind::nverts][geo::ndim];
        Real vert_ref[face_kind::nverts][geo::ndim];
        Index vert_idxs[face_kind::nverts];
        for (int i=0; i<face_kind::nverts; ++i) {
          vert_idxs[i] = face_verts(face_idx,i);
          for (int j=0; j<geo::ndim; ++j) {
            vert_phys[i][j] = phys_crds_verts(vert_idxs[i], j);
            vert_ref[i][j] = ref_crds_verts(vert_idxs[i], j);
          }
        }

        // shift (arbitrary choice)
        // so that vertex 1's coordinates define the origin
        for (int i=0; i<2; ++i) {
          face_ref[i] -= vert_ref[1][i];
          face_phys[i] -= vert_phys[1][i];
        }
        for (int i=0; i<face_kind::nverts; ++i) {
          for (int j=0; j<2; ++j) {
            vert_phys[i][j] -= vert_phys[1][j];
            vert_ref[i][j] -= vert_ref[1][j];
          }
        }
        // let edge 1 define the local x direction
        // its origin is vertex 1, its destination is vertex 2
        const Real edge1_ref[2] = {vert_ref[2][0] - vert_ref[1][0],
                                   vert_ref[2][1] - vert_ref[1][1]};
        const Real edge1_phys[2]= {vert_phys[2][0] - vert_phys[1][0],
                                   vert_phys[2][1] - vert_phys[1][1]};
        const Real dx0 = PlaneGeometry::mag(edge1_ref);
        // build the x unit vector
        Real xdir_ref[2];
        Real xdir_len =0;
        for (int i=0; i<2; ++i) {
          xdir_ref[i] = edge1_ref[i];
          xdir_len += square(edge1_ref[i]);
        }
        xdir_len = sqrt(xdir_len);
        for (int i=0; i<2; ++i) {
          xdir_ref[i] /= xdir_len;
        }

        // for planar quadrilaterals, the "y" direction
        // defined by edge0's reverse direction is already orthogonal to "x"
        // in its reference configuration
        const Real edge0_rev_phys[2] = {vert_phys[0][0], vert_phys[0][1]};
        const Real edge0_rev_ref[2] =  {vert_ref[0][0], vert_ref[0][1]};
        Real ydir_ref[2];
        Real ydir_len = 0;
        const Real dy0 = PlaneGeometry::mag(edge0_rev_ref);
        for (int i=0; i<2; ++i) {
          ydir_ref[i] = edge0_rev_ref[i];
          ydir_len += square(edge0_rev_ref[i]);
        }
        ydir_len = sqrt(ydir_len);
        for (int i=0; i<2; ++i) {
          ydir_ref[i] /= ydir_len;
        }

#ifndef NDEBUG
        const Real x_dot_y = PlaneGeometry::dot(xdir_ref, ydir_ref);
        if (!FloatingPoint<Real>::zero(x_dot_y, fp_tol)) {
          spdlog::warn("x_dot_y = {}", x_dot_y);
        }
#endif
//         LPM_KERNEL_ASSERT(FloatingPoint<Real>::zero(
//           PlaneGeometry::dot(xdir_ref, ydir_ref), fp_tol));

        Real flow_map_gradient[4];
        set_flow_map_gradient(flow_map_gradient, edge0_rev_phys, edge1_phys, xdir_ref, ydir_ref, dx0, dy0);
        Real cg_tensor[4];
        cauchy_green_tensor(cg_tensor, flow_map_gradient);
        const auto eigs = two_by_two_real_eigenvalues(cg_tensor);
        const Real lambda1 = eigs[0];
        const Real lambda2 = eigs[1];
#ifndef NDEBUG
        if (!FloatingPoint<Real>::equiv(lambda1*lambda2, 1, fp_tol)) {
          spdlog::warn("ftle error: abs(lambda1 * lambda2 - 1)= {} (should be 0)",
            abs(lambda1*lambda2-1));
        }
#endif
//         LPM_KERNEL_ASSERT(FloatingPoint<Real>::equiv(lambda1*lambda2, 1, fp_tol));
        // TODO: divide by time here?
        ftle(face_idx) = log(lambda1);// * FloatingPoint<Real>::safe_denominator(2*t);
      }
    }
};

inline Real get_max_ftle(const scalar_view_type ftle, const mask_view_type mask, const Index& nfaces) {
  Real result;
  Kokkos::parallel_reduce(nfaces,
    KOKKOS_LAMBDA (const Index i, Real& m) {
      if (!mask(i)) {
        m = (m > ftle(i) ? m : ftle(i));
      }
    }, Kokkos::Max<Real>(result));
  return result;
}



} // namespace Lpm

#endif
