#ifndef LPM_BIVAR_REMESH_IMPL_HPP
#define LPM_BIVAR_REMESH_IMPL_HPP

#include "LpmConfig.h"
#include "lpm_bivar_remesh.hpp"
#include "mesh/lpm_refinement.hpp"
#include "mesh/lpm_refinement_flags.hpp"
#include "util/lpm_stl_utils.hpp"

namespace Lpm {

template <typename SeedType>
BivarRemesh<SeedType>::BivarRemesh(PolyMesh2d<SeedType>& new_mesh,
    vert_scalar_field_map& new_vert_scalars,
    face_scalar_field_map& new_face_scalars,
    vert_vector_field_map& new_vert_vectors,
    face_vector_field_map& new_face_vectors,
    const PolyMesh2d<SeedType>& old_mesh,
    const vert_scalar_field_map& old_vert_scalars,
    const face_scalar_field_map& old_face_scalars,
    const vert_vector_field_map& old_vert_vectors,
    const face_vector_field_map& old_face_vectors,
    const std::shared_ptr<spdlog::logger> login) :
    new_mesh(new_mesh),
    old_mesh(old_mesh),
    old_vert_scalars(old_vert_scalars),
    old_face_scalars(old_face_scalars),
    old_vert_vectors(old_vert_vectors),
    old_face_vectors(old_face_vectors),
    new_vert_scalars(new_vert_scalars),
    new_face_scalars(new_face_scalars),
    new_vert_vectors(new_vert_vectors),
    new_face_vectors(new_face_vectors),
    logger(login)
  {
    old_gather = std::make_unique<GatherMeshData<SeedType>>(old_mesh);
    old_gather->unpack_coordinates();
    old_gather->init_scalar_fields(old_vert_scalars, old_face_scalars);
    old_gather->init_vector_fields(old_vert_vectors, old_face_vectors);
    old_gather->gather_scalar_fields(old_vert_scalars, old_face_scalars);
    old_gather->gather_vector_fields(old_vert_vectors, old_face_vectors);

    new_gather = std::make_unique<GatherMeshData<SeedType>>(new_mesh);
    new_gather->unpack_coordinates();
    new_gather->init_scalar_fields(new_vert_scalars, new_face_scalars);
    new_gather->init_vector_fields(new_vert_vectors, new_face_vectors);

    new_scatter = std::make_unique<ScatterMeshData<SeedType>>(*new_gather, new_mesh);

    build_in_out_maps();

    bivar = std::make_unique<BivarInterface<SeedType>>(*old_gather,
      new_gather->h_x, new_gather->h_y,
      scalar_in_out_map, vector_in_out_map);

    if (!logger) {
      logger = lpm_logger();
    }
//     logger->debug("old_gather info: {}", old_gather->info_string());
//     logger->debug("new_gather info: {}", new_gather->info_string());
  }

template <typename SeedType>
void BivarRemesh<SeedType>::build_in_out_maps() {
  for (const auto& named_field : old_vert_scalars) {
    LPM_ASSERT(map_contains(old_face_scalars, named_field.first));
    LPM_ASSERT(map_contains(new_vert_scalars, named_field.first));
    LPM_ASSERT(map_contains(new_face_scalars, named_field.first));

    scalar_in_out_map.emplace(named_field.first, named_field.first);
  }
  for (const auto& named_field : old_vert_vectors) {
    LPM_ASSERT(map_contains(old_face_vectors, named_field.first));
    LPM_ASSERT(map_contains(new_vert_vectors, named_field.first));
    LPM_ASSERT(map_contains(new_face_vectors, named_field.first));

    vector_in_out_map.emplace(named_field.first, named_field.first);
  }
}

template <typename SeedType>
void BivarRemesh<SeedType>::uniform_direct_remesh() {
  bivar->interpolate_lag_crds(new_gather->h_lag_crds);
  bivar->interpolate(new_gather->h_scalar_fields, new_gather->h_vector_fields);

  new_gather->update_device();

  new_scatter->scatter_lag_crds();
  new_scatter->scatter_fields(new_vert_scalars, new_face_scalars,
    new_vert_vectors, new_face_vectors);

  logger->debug("uniform direct remesh complete.");
}

template <typename SeedType> template <typename FlagType>
void BivarRemesh<SeedType>::adaptive_direct_remesh(Refinement<SeedType>& refiner,
  const FlagType& flag) {

  uniform_direct_remesh();

  Index vert_start_idx = 0;
  Index face_start_idx = 0;
  for (int i=0; i<new_mesh.params.amr_limit; ++i) {
    const Index vert_end_idx = new_mesh.n_vertices_host();
    const Index face_end_idx = new_mesh.n_faces_host();

    refiner.iterate(face_start_idx, face_end_idx, flag);
    new_mesh.divide_flagged_faces(refiner.flags, *logger);

    new_gather.reset(new GatherMeshData<SeedType>(new_mesh));
    new_gather->unpack_coordinates();
    new_gather->update_host();
    new_gather->init_scalar_fields(new_vert_scalars, new_face_scalars);
    new_gather->init_vector_fields(new_vert_vectors, new_face_vectors);

    bivar.reset(new BivarInterface<SeedType>(*old_gather,
      new_gather->h_x, new_gather->h_y,
      scalar_in_out_map, vector_in_out_map));
    bivar->interpolate_lag_crds(new_gather->h_lag_crds);
    bivar->interpolate(new_gather->h_scalar_fields, new_gather->h_vector_fields);

    new_gather->update_device();

    new_scatter.reset(new ScatterMeshData<SeedType>(*new_gather, new_mesh));
    new_scatter->scatter_lag_crds();
    new_scatter->scatter_fields(new_vert_scalars, new_face_scalars,
    new_vert_vectors, new_face_vectors);

    vert_start_idx = vert_end_idx;
    face_start_idx = face_end_idx;
  }
}

template <typename SeedType> template <typename FlagType1, typename FlagType2>
void BivarRemesh<SeedType>::adaptive_direct_remesh(Refinement<SeedType>& refiner,
  const FlagType1& flag1, const FlagType2& flag2) {

  uniform_direct_remesh();

  Index vert_start_idx = 0;
  Index face_start_idx = 0;
  for (int i=0; i<new_mesh.params.amr_limit; ++i) {
    const Index vert_end_idx = new_mesh.n_vertices_host();
    const Index face_end_idx = new_mesh.n_faces_host();

    refiner.iterate(face_start_idx, face_end_idx, flag1, flag2);
    logger->debug("amr iter. {}: flag1 count {}, flag2 count {}.",
      i, refiner.count[0], refiner.count[1]);

    new_mesh.divide_flagged_faces(refiner.flags, *logger);

    new_gather.reset(new GatherMeshData<SeedType>(new_mesh));
    new_gather->unpack_coordinates();
    new_gather->update_host();
    new_gather->init_scalar_fields(new_vert_scalars, new_face_scalars);
    new_gather->init_vector_fields(new_vert_vectors, new_face_vectors);

    bivar.reset(new BivarInterface<SeedType>(*old_gather,
      new_gather->h_x, new_gather->h_y,
      scalar_in_out_map, vector_in_out_map));

    bivar->interpolate_lag_crds(new_gather->h_lag_crds);
    bivar->interpolate(new_gather->h_scalar_fields, new_gather->h_vector_fields);

    new_gather->update_device();

    new_scatter.reset(new ScatterMeshData<SeedType>(*new_gather, new_mesh));
    new_scatter->scatter_lag_crds();
    new_scatter->scatter_fields(new_vert_scalars, new_face_scalars,
    new_vert_vectors, new_face_vectors);

    vert_start_idx = vert_end_idx;
    face_start_idx = face_end_idx;
  }
}

template <typename SeedType> template <typename FlagType1, typename FlagType2, typename FlagType3>
void BivarRemesh<SeedType>::adaptive_direct_remesh(Refinement<SeedType>& refiner,
  const FlagType1& flag1, const FlagType2& flag2, const FlagType3& flag3) {

  uniform_direct_remesh();

  Index vert_start_idx = 0;
  Index face_start_idx = 0;
  for (int i=0; i<new_mesh.params.amr_limit; ++i) {
    const Index vert_end_idx = new_mesh.n_vertices_host();
    const Index face_end_idx = new_mesh.n_faces_host();

    refiner.iterate(face_start_idx, face_end_idx, flag1, flag2, flag3);
    logger->debug("amr iter. {}: flag1 count {}, flag2 count {}.",
      i, refiner.count[0], refiner.count[1]);

    new_mesh.divide_flagged_faces(refiner.flags, *logger);

    new_gather.reset(new GatherMeshData<SeedType>(new_mesh));
    new_gather->unpack_coordinates();
    new_gather->update_host();
    new_gather->init_scalar_fields(new_vert_scalars, new_face_scalars);
    new_gather->init_vector_fields(new_vert_vectors, new_face_vectors);

    bivar.reset(new BivarInterface<SeedType>(*old_gather,
      new_gather->h_x, new_gather->h_y,
      scalar_in_out_map, vector_in_out_map));

    bivar->interpolate_lag_crds(new_gather->h_lag_crds);
    bivar->interpolate(new_gather->h_scalar_fields, new_gather->h_vector_fields);

    new_gather->update_device();

    new_scatter.reset(new ScatterMeshData<SeedType>(*new_gather, new_mesh));
    new_scatter->scatter_lag_crds();
    new_scatter->scatter_fields(new_vert_scalars, new_face_scalars,
    new_vert_vectors, new_face_vectors);

    vert_start_idx = vert_end_idx;
    face_start_idx = face_end_idx;
  }
}


template <typename SeedType> template <typename VorticityFunctor>
void BivarRemesh<SeedType>::uniform_indirect_remesh(const VorticityFunctor& vorticity,
  const CoriolisBetaPlane& coriolis) {
  bivar->interpolate_lag_crds(new_gather->h_lag_crds);
  bivar->interpolate_vectors(new_gather->h_vector_fields);
  new_gather->update_device();
  auto lag_crd_view = new_gather->lag_crds;
  auto phys_crd_view = new_gather->phys_crds;
  auto abs_vort_view = new_gather->scalar_fields.at("absolute_vorticity");
  auto rel_vort_view = new_gather->scalar_fields.at("relative_vorticity");
  Kokkos::parallel_for(new_gather->n(),
    KOKKOS_LAMBDA (const Index i) {
      const auto mpcrd = Kokkos::subview(phys_crd_view, i, Kokkos::ALL);
      const auto mlcrd = Kokkos::subview(lag_crd_view, i, Kokkos::ALL);
      const Real omega = vorticity(mlcrd) + coriolis.f(mlcrd);
      abs_vort_view(i) = omega;
      rel_vort_view(i) = omega - coriolis.f(mpcrd);
    });
  new_scatter->scatter_lag_crds();
  new_scatter->scatter_fields(new_vert_scalars, new_face_scalars,
    new_vert_vectors, new_face_vectors);
}

template <typename SeedType> template <typename VorticityFunctor, typename Tracer1>
void BivarRemesh<SeedType>::uniform_indirect_remesh(const VorticityFunctor& vorticity,
  const CoriolisBetaPlane& coriolis, const Tracer1& tracer1) {
  bivar->interpolate_lag_crds(new_gather->h_lag_crds);
  bivar->interpolate_vectors(new_gather->h_vector_fields);
  new_gather->update_device();
  auto lag_crd_view = new_gather->lag_crds;
  auto phys_crd_view = new_gather->phys_crds;
  auto abs_vort_view = new_gather->scalar_fields.at("absolute_vorticity");
  auto rel_vort_view = new_gather->scalar_fields.at("relative_vorticity");
  auto tracer_view1 = new_gather->scalar_fields.at(tracer1.name());
  Kokkos::parallel_for(new_gather->n(),
    KOKKOS_LAMBDA (const Index i) {
      const auto mpcrd = Kokkos::subview(phys_crd_view, i, Kokkos::ALL);
      const auto mlcrd = Kokkos::subview(lag_crd_view, i, Kokkos::ALL);
      const Real omega = vorticity(mlcrd) + coriolis.f(mlcrd);
      abs_vort_view(i) = omega;
      rel_vort_view(i) = omega - coriolis.f(mpcrd);
      tracer_view1(i) = tracer1(mlcrd);
    });
  new_scatter->scatter_lag_crds();
  new_scatter->scatter_fields(new_vert_scalars, new_face_scalars,
    new_vert_vectors, new_face_vectors);
}

template <typename SeedType> template <typename VorticityFunctor, typename Tracer1, typename Tracer2>
void BivarRemesh<SeedType>::uniform_indirect_remesh(const VorticityFunctor& vorticity,
  const CoriolisBetaPlane& coriolis, const Tracer1& tracer1,
  const Tracer2& tracer2) {
  bivar->interpolate_lag_crds(new_gather->h_lag_crds);
  bivar->interpolate_vectors(new_gather->h_vector_fields);
  new_gather->update_device();
  auto lag_crd_view = new_gather->lag_crds;
  auto phys_crd_view = new_gather->phys_crds;
  auto abs_vort_view = new_gather->scalar_fields.at("absolute_vorticity");
  auto rel_vort_view = new_gather->scalar_fields.at("relative_vorticity");
  auto tracer_view1 = new_gather->scalar_fields.at(tracer1.name());
  auto tracer_view2 = new_gather->scalar_fields.at(tracer2.name());
  Kokkos::parallel_for(new_gather->n(),
    KOKKOS_LAMBDA (const Index i) {
      const auto mpcrd = Kokkos::subview(phys_crd_view, i, Kokkos::ALL);
      const auto mlcrd = Kokkos::subview(lag_crd_view, i, Kokkos::ALL);
      const Real omega = vorticity(mlcrd) + coriolis.f(mlcrd);
      abs_vort_view(i) = omega;
      rel_vort_view(i) = omega - coriolis.f(mpcrd);
      tracer_view1(i) = tracer1(mlcrd);
      tracer_view2(i) = tracer2(mlcrd);
    });
  new_scatter->scatter_lag_crds();
  new_scatter->scatter_fields(new_vert_scalars, new_face_scalars,
    new_vert_vectors, new_face_vectors);
}

template <typename SeedType>
template <typename VorticityFunctor, typename RefinerType, typename FlagType>
void BivarRemesh<SeedType>::adaptive_indirect_remesh(const VorticityFunctor& vorticity,
    const CoriolisBetaPlane& coriolis,
    RefinerType& refiner, const FlagType& flag) {

  uniform_indirect_remesh(vorticity, coriolis);

  Index vert_start_idx = 0;
  Index face_start_idx = 0;
  for (int i=0; i<new_mesh.params.amr_limit; ++i) {
    const Index vert_end_idx = new_mesh.n_vertices_host();
    const Index face_end_idx = new_mesh.n_faces_host();

    refiner.iterate(face_start_idx, face_end_idx, flag);
    logger->debug("amr iter. {}: flag1 count {}", i, refiner.count[0]);

    new_mesh.divide_flagged_faces(refiner.flags, *logger);

    new_gather.reset(new GatherMeshData<SeedType>(new_mesh));
    new_gather->unpack_coordinates();
    new_gather->update_host();
    new_gather->init_scalar_fields(new_vert_scalars, new_face_scalars);
    new_gather->init_vector_fields(new_vert_vectors, new_face_vectors);

    bivar.reset(new BivarInterface<SeedType>(*old_gather,
      new_gather->h_x, new_gather->h_y,
      scalar_in_out_map, vector_in_out_map));
    new_scatter.reset(new ScatterMeshData<SeedType>(*new_gather, new_mesh));

    uniform_indirect_remesh(vorticity, coriolis);

    vert_start_idx = vert_end_idx;
    face_start_idx = face_end_idx;
  }
}

template <typename SeedType>
template <typename VorticityFunctor, typename RefinerType,
          typename FlagType1, typename FlagType2>
void BivarRemesh<SeedType>::adaptive_indirect_remesh(const VorticityFunctor& vorticity,
    const CoriolisBetaPlane& coriolis,
    RefinerType& refiner, const FlagType1& flag1, const FlagType2& flag2) {

  uniform_indirect_remesh(vorticity, coriolis);

  Index vert_start_idx = 0;
  Index face_start_idx = 0;
  for (int i=0; i<new_mesh.params.amr_limit; ++i) {
    const Index vert_end_idx = new_mesh.n_vertices_host();
    const Index face_end_idx = new_mesh.n_faces_host();

    refiner.iterate(face_start_idx, face_end_idx, flag1, flag2);
    logger->debug("amr iter. {}: flag1 count {}", i, refiner.count[0]);
    logger->debug("amr iter. {}: flag2 count {}", i, refiner.count[1]);

    new_mesh.divide_flagged_faces(refiner.flags, *logger);

    new_gather.reset(new GatherMeshData<SeedType>(new_mesh));
    new_gather->unpack_coordinates();
    new_gather->update_host();
    new_gather->init_scalar_fields(new_vert_scalars, new_face_scalars);
    new_gather->init_vector_fields(new_vert_vectors, new_face_vectors);

    bivar.reset(new BivarInterface<SeedType>(*old_gather,
      new_gather->h_x, new_gather->h_y,
      scalar_in_out_map, vector_in_out_map));
    new_scatter.reset(new ScatterMeshData<SeedType>(*new_gather, new_mesh));

    uniform_indirect_remesh(vorticity, coriolis);

    vert_start_idx = vert_end_idx;
    face_start_idx = face_end_idx;
  }
}

template <typename SeedType>
template <typename VorticityFunctor, typename RefinerType,
          typename FlagType1, typename FlagType2, typename FlagType3>
void BivarRemesh<SeedType>::adaptive_indirect_remesh(const VorticityFunctor& vorticity,
    const CoriolisBetaPlane& coriolis,
    RefinerType& refiner, const FlagType1& flag1, const FlagType2& flag2,
    const FlagType3& flag3) {

  uniform_indirect_remesh(vorticity, coriolis);

  Index vert_start_idx = 0;
  Index face_start_idx = 0;
  for (int i=0; i<new_mesh.params.amr_limit; ++i) {
    const Index vert_end_idx = new_mesh.n_vertices_host();
    const Index face_end_idx = new_mesh.n_faces_host();

    refiner.iterate(face_start_idx, face_end_idx, flag1, flag2, flag3);
    logger->debug("amr iter. {}: flag1 count {}", i, refiner.count[0]);
    logger->debug("amr iter. {}: flag2 count {}", i, refiner.count[1]);
    logger->debug("amr iter. {}: flag3 count {}", i, refiner.count[2]);

    new_mesh.divide_flagged_faces(refiner.flags, *logger);

    new_gather.reset(new GatherMeshData<SeedType>(new_mesh));
    new_gather->unpack_coordinates();
    new_gather->update_host();
    new_gather->init_scalar_fields(new_vert_scalars, new_face_scalars);
    new_gather->init_vector_fields(new_vert_vectors, new_face_vectors);

    bivar.reset(new BivarInterface<SeedType>(*old_gather,
      new_gather->h_x, new_gather->h_y,
      scalar_in_out_map, vector_in_out_map));
    new_scatter.reset(new ScatterMeshData<SeedType>(*new_gather, new_mesh));

    uniform_indirect_remesh(vorticity, coriolis);

    vert_start_idx = vert_end_idx;
    face_start_idx = face_end_idx;
  }
}

} // namespace Lpm

#endif
