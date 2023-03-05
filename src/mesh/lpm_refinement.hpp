#ifndef LPM_REFINEMENT_HPP
#define LPM_REFINEMENT_HPP

#include "LpmConfig.h"
#include "lpm_assert.hpp"
#include "lpm_field.hpp"
#include "lpm_geometry.hpp"
#include "lpm_logger.hpp"
#include "mesh/lpm_polymesh2d.hpp"

namespace Lpm {

typedef Kokkos::View<bool*> flag_view;

/** @brief Given a relative tolerance in [0,1], convert to an absolute tolerance that accounts for mesh size.
Enables the same tolerance values to be used even as the background uniform mesh is refined.
*/
inline Real convert_to_absolute_tol(const Real rel_tol, const Real max_val) {
  return rel_tol * max_val;
}

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
      flag_view f,
      const std::shared_ptr<const PolyMesh2d<MeshSeedType>> mesh,
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
        min_lag_crds[j] = vertex_lag_crds(face_vertex_view(j), 0);
        max_lag_crds[j] = vertex_lag_crds(face_vertex_view(j), 0);
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

      flags(i) = ( flags(i) or (dsum > tol) );
    }
  }
};

struct ScalarIntegralFlag {
  flag_view flags;
  scalar_view_type face_vals;
  scalar_view_type area;
  mask_view_type facemask;
  Real tol;

  ScalarIntegralFlag(flag_view f, const scalar_view_type fv,
   const scalar_view_type a, const mask_view_type m, const Real eps)
      : flags(f), face_vals(fv), area(a), facemask(m), tol(eps) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index i) const {
    if (!facemask(i)) {
      flags(i) = ( flags(i) or (face_vals(i) * area(i) > tol) );
    }
  }
};

struct ScalarVariationFlag {
  flag_view flags;
  scalar_view_type face_vals;
  scalar_view_type vert_vals;
  Kokkos::View<Index**> face_vertex_view;
  mask_view_type facemask;
  Real tol;

  ScalarVariationFlag(flag_view f, const scalar_view_type fv,
    const scalar_view_type vv, const Kokkos::View<Index**> verts,
    const mask_view_type m, const Real eps) :
    flags(f), face_vals(fv), vert_vals(vv), face_vertex_view(verts),
    facemask(m), tol(eps) {}

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
      flags(i) = ( flags(i) or (maxval - minval > tol));
    }
  }
};


}  // namespace Lpm

#endif
