#ifndef LPM_SCATTER_MESH_DATA_IMPL_HPP
#define LPM_SCATTER_MESH_DATA_IMPL_HPP

#include "mesh/lpm_scatter_mesh_data.hpp"

namespace Lpm {

template <typename SeedType>
ScatterMeshData<SeedType>::ScatterMeshData(const GatherMeshData<SeedType>& out,
  const std::shared_ptr<PolyMesh2d<SeedType>> pm) :
  output(out),
  mesh(pm)
{}

template <typename SeedType>
void ScatterMeshData<SeedType>::scatter(
    const std::map<std::string, ScalarField<VertexField>> vertex_scalar_fields,
    const std::map<std::string, ScalarField<FaceField>> face_scalar_fields,
    const std::map<std::string,
      VectorField<typename SeedType::geo, VertexField>>& vertex_vector_fields,
    const std::map<std::string,
      VectorField<typename SeedType::geo, FaceField>>& face_vector_fields)
{
  const auto face_mask = mesh->faces.mask;
  const auto face_leaf_idx = mesh->faces.leaf_idx;
  const Index n_verts = mesh->n_vertices_host();
  const Index n_faces = mesh->n_faces_host();

  for (const auto& sf : vertex_scalar_fields) {
    LPM_REQUIRE( face_scalar_fields.find(sf.first) != face_scalar_fields.end() );
    if ( output.scalar_fields.find(sf.first) != output.scalar_fields.end() ) {

      auto mesh_vert_view = sf.second.view;
      auto mesh_face_view = face_scalar_fields.at(sf.first).view;
      const auto src_view = output.scalar_fields.at(sf.first);
      Kokkos::parallel_for(n_verts,
        KOKKOS_LAMBDA (const Index i) {
          mesh_vert_view(i) = src_view(i);
        });
      Kokkos::parallel_for(n_faces,
        KOKKOS_LAMBDA (const Index i) {
          if (!face_mask(i)) {
            mesh_face_view(i) = src_view(n_verts + face_leaf_idx(i));
          }
        });
    }
  }

  for (const auto& vf : vertex_vector_fields) {
    LPM_REQUIRE( face_vector_fields.find(vf.first) !=
                 face_vector_fields.end() );
    if ( output.vector_fields.find(vf.first) != output.vector_fields.end() ) {
      auto mesh_vert_view = vf.second.view;
      auto mesh_face_view = face_vector_fields.at(vf.first).view;
      const auto src_view = output.vector_fields.at(vf.first);
      Kokkos::parallel_for(n_verts,
        KOKKOS_LAMBDA (const Index i) {
          for (Int j=0; j<SeedType::geo::ndim; ++j) {
            mesh_vert_view(i,j) = src_view(i,j);
          }
        });
      Kokkos::parallel_for(n_faces,
        KOKKOS_LAMBDA (const Index i) {
          if (!face_mask(i)) {
            for (Int j=0; j<SeedType::geo::ndim; ++j) {
              mesh_face_view(i,j) = src_view(n_verts + face_leaf_idx(i),j);
            }
          }
        });
    }
  }
}

} // namespace Lpm

#endif
