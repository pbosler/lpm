#ifndef LPM_REFINEMENT_FLAGS_HPP
#define LPM_REFINEMENT_FLAGS_HPP

#include "LpmConfig.h"

namespace Lpm {

/** All refinement flag functors toggle a flag view entry to True
if their condition is met.

They do not change an existing True in that same entry.
*/

/** A flag view is 1-1 with a mesh's panels.

  flag_view(i) = true will trigger refinement of that panel in an
          adaptive refinement workflow.
  flag_view(i) = false will not.
*/
typedef Kokkos::View<bool*> flag_view;

template <typename MeshSeedType>
struct FlowMapVariationFlag {
  typedef typename MeshSeedType::geo::crd_view_type crd_view_type;

  flag_view flags;
  crd_view_type vertex_lag_crds;
  Kokkos::View<Index**> face_vertex_view;
  mask_view_type facemask;
  Real tol;
  static constexpr Int nverts = MeshSeedType::faceKind::nverts;

  FlowMapVariationFlag(
      flag_view f, const std::shared_ptr<const PolyMesh2d<MeshSeedType>> mesh,
      const Real tol_)
      : flags(f),
        vertex_lag_crds(mesh->vertices.lag_crds.crds),
        face_vertex_view(mesh->faces.verts),
        facemask(mesh->faces.mask),
        tol(tol_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index i) const {
    if (!facemask(i)) {
      Real min_lag_crds[MeshSeedType::geo::ndim];
      Real max_lag_crds[MeshSeedType::geo::ndim];
      for (int j = 0; j < MeshSeedType::geo::ndim; ++j) {
        min_lag_crds[j] = vertex_lag_crds(face_vertex_view(i, 0), j);
        max_lag_crds[j] = vertex_lag_crds(face_vertex_view(i, 0), j);
      }

      for (int j = 1; j < nverts; ++j) {
        const auto x0 = Kokkos::subview(vertex_lag_crds, face_vertex_view(i, j),
                                        Kokkos::ALL);
        for (int k = 0; k < MeshSeedType::geo::ndim; ++k) {
          if (x0[k] < min_lag_crds[k]) min_lag_crds[k] = x0[k];
          if (x0[k] > max_lag_crds[k]) max_lag_crds[k] = x0[k];
        }
      }

      Real dsum = 0;
      for (int j = 0; j < MeshSeedType::geo::ndim; ++j) {
        dsum += max_lag_crds[j] - min_lag_crds[j];
      }

      flags(i) = (flags(i) or (dsum > tol));
    }
  }
};

struct ScalarMaxFlag {
  flag_view flags;
  scalar_view_type face_vals;
  mask_view_type facemask;
  Index nfaces;
  Real tol;

  ScalarMaxFlag(flag_view f, const scalar_view_type fv,
    const mask_view_type m, const Index n, const Real eps) :
    flags(f), face_vals(fv), facemask(m), nfaces(n), tol(eps) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index i) const {
    if (not facemask(i) ) {
      flags(i) = ( flags(i) or ( abs(face_vals(i)) > tol ) );
    }
  }
};

struct ScalarIntegralFlag {
  flag_view flags;
  scalar_view_type face_vals;
  scalar_view_type area;
  mask_view_type facemask;
  Index nfaces;
  Real tol;


  ScalarIntegralFlag(flag_view f, const scalar_view_type fv,
                     const scalar_view_type a, const mask_view_type m,
                     const Index n,
                     const Real eps)
      : flags(f), face_vals(fv), area(a), facemask(m), nfaces(n), tol(eps) {}

  struct MaxAbsValTag {};

  void set_tol_from_relative_value() {
    Real max_abs;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<MaxAbsValTag>(0, nfaces),
    *this, Kokkos::Max<Real>(max_abs));
    tol *= max_abs;
  }

  /// Kernel for parallel reduce.
  /// Returns the maximum absolute value of the flag-associated quantity.
  KOKKOS_INLINE_FUNCTION
  void operator () (MaxAbsValTag, const Index i, Real& m) const {
    if (!facemask(i)) {
      const Real val = abs(face_vals(i)) * area(i);
      m = (val > m ? val : m);
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index i) const {
    if (!facemask(i)) {
      flags(i) = (flags(i) or (abs(face_vals(i)) * area(i) > tol));
    }
  }
};

struct ScalarVariationFlag {
  flag_view flags;
  scalar_view_type face_vals;
  scalar_view_type vert_vals;
  Kokkos::View<Index**> face_vertex_view;
  mask_view_type facemask;
  Index nfaces;
  Real tol;

  ScalarVariationFlag(flag_view f, const scalar_view_type fv,
                      const scalar_view_type vv,
                      const Kokkos::View<Index**> verts, const mask_view_type m,
                      const Index n,
                      const Real eps)
      : flags(f),
        face_vals(fv),
        vert_vals(vv),
        face_vertex_view(verts),
        facemask(m),
        nfaces(n),
        tol(eps) {}

  struct MaxAbsValTag {};

  void set_tol_from_relative_value() {
    Real max_abs;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<MaxAbsValTag>(0, nfaces),
    *this, Kokkos::Max<Real>(max_abs));
    tol *= max_abs;
  }

  /// Kernel for parallel reduce.
  /// Returns the maximum absolute value of the flag-associated quantity.
  KOKKOS_INLINE_FUNCTION
  void operator() (MaxAbsValTag, const Index i, Real& m) const {
    if (!facemask(i)) {
      Real minval = face_vals(i);
      Real maxval = face_vals(i);
      for (int j=0; j<face_vertex_view.extent(1); ++j) {
        const auto v_idx = face_vertex_view(i,j);
        if (vert_vals(v_idx) < minval) minval = vert_vals(v_idx);
        if (vert_vals(v_idx) > maxval) maxval = vert_vals(v_idx);
      }
      Real dval = maxval - minval;
      m = (dval > m ? dval : m);
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index i) const {
    if (!facemask(i)) {
      Real minval = face_vals(i);
      Real maxval = face_vals(i);
      for (int j = 0; j < face_vertex_view.extent(1); ++j) {
        const auto vidx = face_vertex_view(i, j);
        if (vert_vals(vidx) < minval) minval = vert_vals(vidx);
        if (vert_vals(vidx) > maxval) maxval = vert_vals(vidx);
      }
      flags(i) = (flags(i) or (maxval - minval > tol));
    }
  }
};

}  // namespace Lpm

#endif
