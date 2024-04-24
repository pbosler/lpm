#ifndef LPM_BIVAR_REMESH_IMPL_HPP
#define LPM_BIVAR_REMESH_IMPL_HPP

#include "LpmConfig.h"
#include "lpm_bivar_remesh.hpp"
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
    const face_vector_field_map& old_face_vectors) :
    new_mesh(new_mesh),
    old_mesh(old_mesh),
    old_vert_scalars(old_vert_scalars),
    old_face_scalars(old_face_scalars),
    old_vert_vectors(old_vert_vectors),
    old_face_vectors(old_face_vectors),
    new_vert_scalars(new_vert_scalars),
    new_face_scalars(new_face_scalars),
    new_vert_vectors(new_vert_vectors),
    new_face_vectors(new_face_vectors)
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
}




} // namespace Lpm

#endif
