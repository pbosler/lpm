#ifndef LPM_BIVAR_REMESH_IMPL_HPP
#define LPM_BIVAR_REMESH_IMPL_HPP

#include "LpmConfig.h"
#include "lpm_bivar_remesh.hpp"

namespace Lpm {

template <typename SeedType>
BivarRemesh<SeedType>::BivarRemesh(PolyMesh2d<SeedType>& new_mesh,
    const PolyMesh2d<SeedType>& old_mesh,
    const std::vector<std::string>& scalar_names, const std::vector<std::string>& vector_names) :
    new_mesh(new_mesh),
    old_mesh(old_mesh)
  {
    old_gather = std::make_unique<GatherMeshData<SeedType>>(old_mesh);
    old_gather->unpack_coordinates();
    new_gather = std::make_unique<GatherMeshData<SeedType>>(new_mesh);
    new_scatter = std::make_unique<ScatterMeshData<SeedType>>(*new_gather, new_mesh);
    bivar = std::make_unique<BivarInterface<SeedType>>(*old_gather,
      new_gather->h_x, new_gather->h_y);
  }

template <typename SeedType>
void BivarRemesh<SeedType>::init_scalar_fields(
  const std::map<std::string, ScalarField<VertexField>>& old_vert_fields,
  const std::map<std::string, ScalarField<FaceField>>& old_face_fields,
  const std::map<std::string, ScalarField<VertexField>>& new_vert_fields,
  const std::map<std::string, ScalarField<FaceField>>& new_face_fields)
{
    old_gather->init_scalar_fields(old_vert_fields, old_face_fields);
    old_gather->gather_scalar_fields(old_vert_fields, old_face_fields);

    in_out_map scalar_in_out_map;
    for (const auto& name : old_vert_fields) {
      scalar_in_out_map.emplace(name, name);
    }
    new_gather->init_scalar_fields(new_vert_fields, new_face_fields);

    bivar.scalar_in_out_map = std::move(scalar_in_out_map);
}

template <typename SeedType>
void BivarRemesh<SeedType>::init_vector_fields(
  const std::map<std::string,
    VectorField<PlaneGeometry, VertexField>>& old_vert_fields,
  const std::map<std::string,
    VectorField<PlaneGeometry, FaceField>>& old_face_fields,
  const std::map<std::string,
    VectorField<PlaneGeometry, VertexField>>& new_vert_fields,
  const std::map<std::string,
    VectorField<PlaneGeometry, FaceField>>& new_face_fields) {

    old_gather->init_vector_fields(old_vert_fields, old_face_fields);
    old_gather->gather_vector_fields(old_vert_fields, old_face_fields);

    new_gather->init_vector_fields(new_vert_fields, new_face_fields);

    in_out_map vector_in_out_map;
    for (const auto& name : old_vert_fields) {
      vector_in_out_map.emplace(name, name);
    }
    bivar.vector_in_out_map = std::move(vector_in_out_map);
}


} // namespace Lpm

#endif
