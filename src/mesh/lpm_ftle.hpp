#ifndef LPM_FTLE_HPP
#define LPM_FTLE_HPP

#include "LpmConfig.h"
#include "lpm_geometry.hpp"
#include "mesh/lpm_mesh_seed.hpp"
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
  static_assert(std::is_same<geo, SphereGeometry>::value,
    "FTLE for planar domains not implemented yet.");

  scalar_view_type ftle; /// output
  crd_view phys_crds_verts; /// input
  crd_view phys_crds_faces; /// input
  crd_view ref_crds_verts; /// input
  crd_view ref_crds_faces; /// input
  face_vertex_view face_verts; /// input

  ComputeFTLE(scalar_view_type& ftle, const crd_view pcv, const crd_view pcf,
    const crd_view rcv, const crd_view rcf,
    const Kokkos::View<Index*[face_kind::nverts] fv) :
    ftle(ftle),
    phys_crds_verts(pcv),
    phys_crds_faces(pcf),
    ref_crds_verts(rcv),
    ref_crds_faces(rfc),
    face_verts(fv) {}

    KOKKOS_INLINE_FUNCTION
    void operator (const Index face_idx) const {
      const auto fai = Kokkos::subview(ref_crds_faces, face_idx, Kokkos::ALL);
      const auto fxi = Kokkos::subview(phys_crds_faces, face_idx, Kokkos::ALL);
      SphereGeometry::normalize(fxi);
      Kokkos::Tuple<Real,9> rot_mat_ref = north_pole_rotation_matrix(fai);
      Kokkos::Tuple<Real,9> rot_mat_phys = north_pole_rotation_matrix(fxi);

      Real vert_phys[face_kind::nverts][geo::ndim];
      Real vert_ref[face_kind::nverts][geo::ndim];
      for (int i=0; i<face_kind::nverts; ++i) {
        const Index vert_idx = face_verts(face_idx, i);
        for (int j=0; j<geo::ndim; ++j) {
          vert_phys[i][j] = phys_crds_verts(vert_idx, j);
          vert_ref[i][j] = ref_crds_verts(vert_idx, j);
        }
      }

      /*

        The rest of the computation goes here...

      */
      ftle(face_idx) = result;
    }

};



} // namespace Lpm

#endif
