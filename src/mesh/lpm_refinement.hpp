#ifndef LPM_REFINEMENT_HPP
#define LPM_REFINEMENT_HPP

#include "LpmConfig.h"
#include "lpm_geometry.hpp"
#include "lpm_field.hpp"
#include "lpm_logger.hpp"
#include "lpm_assert.hpp"
#include "mesh/lpm_polymesh2d.hpp"

namespace Lpm {

template <typename MeshSeedType>
struct FlowMapVariationFlagFunctor {
  typedef typename MeshSeedType::geo::crd_view_type crd_view_type;

  Kokkos::View<bool*> flags;
  crd_view_type vertex_lag_crds;
  Kokkos::View<Index**> face_vertex_view;
  mask_view_type facemask;
  Real tol;
  static constexpr Int nverts = MeshSeedType::faceKind::nverts;

  FlowMapVariationFlagFunctor(Kokkos::View<bool*> f,
    const std::shared_ptr<const PolyMesh2d<MeshSeedType>> mesh,
    const Real tol_) :
  flags(f),
  vertex_lag_crds(mesh->vertices.lag_crds.crds),
  face_vertex_view(mesh->faces.verts),
  facemask(mesh->faces.mask),
  tol(tol_) {}


  KOKKOS_INLINE_FUNCTION
  void operator() (const Index i) const {
    if (!flags(i) and !facemask(i)) {
      Real min_lag_crds[MeshSeedType::geo::ndim];
      Real max_lag_crds[MeshSeedType::geo::ndim];
      for (int j=0; j<MeshSeedType::geo::ndim; ++j) {
        min_lag_crds[j] = vertex_lag_crds(face_vertex_view(j),0);
        max_lag_crds[j] = vertex_lag_crds(face_vertex_view(j),0);
      }

      for (int j=1; j<nverts; ++j) {
        const auto x0 = Kokkos::subview(vertex_lag_crds, face_vertex_view(i,j), Kokkos::ALL);
        for (int k=0; k<MeshSeedType::geo::ndim; ++k) {
          if (x0[k] < min_lag_crds[k]) min_lag_crds[k] = x0[k];
          if (x0[k] > max_lag_crds[k]) max_lag_crds[k] = x0[k];
        }
      }

      Real dsum = 0;
      for (int j=0; j<MeshSeedType::geo::ndim; ++j) {
        dsum += max_lag_crds[j] - min_lag_crds[j];
      }

      flags(i) = (dsum > tol);
    }
  }
};

struct ScalarIntegralFlagFunctor {
  Kokkos::View<bool*> flags;
  scalar_view_type face_vals;
  scalar_view_type area;
  mask_view_type facemask;
  Real tol;

  ScalarIntegralFlagFunctor(Kokkos::View<bool*> f, const scalar_view_type fv,
    const scalar_view_type a, const Real eps) :
    flags(f),
    face_vals(fv),
    area(a),
    tol(eps) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index i) const {
    if (!flags(i) and !facemask(i)) {
      flags(i) = (face_vals(i) * area(i) > tol);
    }
  }
};

struct ScalarVariationFlagFunctor {
  Kokkos::View<bool*> flags;
  scalar_view_type face_vals;
  scalar_view_type vert_vals;
  Kokkos::View<Index**> face_vertex_view;
  mask_view_type facemask;

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index i) const {
    if (!flags(i) and !facemask(i)) {
      Real minval = face_vals(i);
      Real maxval = face_vals(i);
      for (int j=0; j<face_vertex_view.extent(1); ++j) {
        const auto vidx = face_vertex_view(i,j);
        if (vert_vals(vidx) < minval) minval = vert_vals(vidx);
        if (vert_vals(vidx) > maxval) maxval = vert_vals(vidx);
      }
      flags(i) = (maxval - minval > tol);
    }
  }
};

template <typename SeedType, typename LoggerType = Logger<>>
void divide_flagged_faces(std::shared_ptr<PolyMesh2d<SeedType>> mesh, const Kokkos::View<bool*> flags, const PolyMeshParameters& params, LoggerType& logger) {

  Index flag_count;
  Kokkos::parallel_reduce(mesh->faces.n_host(), KOKKOS_LAMBDA (const Index i, Index& s) {
    s += (flags(i) : 1 : 0);
  }, flag_count);
  const Index space_left = params.nmaxfaces - mesh->faces.n_host();

  if (flag_count > space_left/4) {
    logger.warn("divide_flagged_faces: not enough memory for AMR (flag_count = {}, nfaces = {}, nmaxfaces = {})",
      flag_count, mesh->faces.n_host(), params.nmaxfaces);
  }
  else {
    const Index n_faces_in = mesh->faces.n_host();
    auto host_flags = Kokkos::create_mirror_view(flags);
    Kokkos::deep_copy(host_flags, flags);
    Index refine_count = 0;
    bool limit_reached = false;
    for (Index i=0; i<n_faces_in; ++i) {
      if (host_flags(i)) {
        if (mesh->faces.host_level(i) < params.init_depth + params.amr_limit) {
          mesh->divide_face(i);
          ++refine_count;
        }
        else {
          limit_reached = true;
        }
      }
    }
    if (limit_reached) {
      logger.warn("divide_flagged_faces: local refinement limit reached; divided {} of {} flagged faces.",
        refine_count, flag_count);
    }
    else {
      LPM_ASSERT(refine_count == flag_count);
      logger.info("divide_flagged_faces: {} faces divided", refine_count);
    }
  }
}

} // namespace Lpm

#endif
