#ifndef LPM_SCATTER_MESH_DATA_IMPL_HPP
#define LPM_SCATTER_MESH_DATA_IMPL_HPP

#include "mesh/lpm_scatter_mesh_data.hpp"

namespace Lpm {

template <typename SeedType>
ScatterMeshData<SeedType>::ScatterMeshData(const GatherMeshData<SeedType>& out,
                                           PolyMesh2d<SeedType>& pm)
    : output(out), mesh(pm) {}

template <typename SeedType>
void ScatterMeshData<SeedType>::scatter_lag_crds() {
  const auto face_mask = mesh.faces.mask;
  const auto face_leaf_idx = mesh.faces.leaf_idx;
  const Index n_verts = mesh.n_vertices_host();
  const Index n_faces = mesh.n_faces_host();

  const auto src_view = output.lag_crds;
  auto v_lag_crds = mesh.vertices.lag_crds.view;
  const auto vert_policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
      {0, 0}, {n_verts, SeedType::geo::ndim});

  Kokkos::parallel_for(
      "scatter_vert_lag_crds", vert_policy,
      KOKKOS_LAMBDA(const Index i, const int j) {
        v_lag_crds(i, j) = src_view(i, j);
      });

  const auto face_policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
      {0, 0}, {n_faces, SeedType::geo::ndim});
  auto f_lag_crds = mesh.faces.lag_crds.view;
  Kokkos::parallel_for(
      "scatter_face_lag_crds", face_policy,
      KOKKOS_LAMBDA(const Index i, const Int j) {
        if (!face_mask(i)) {
          f_lag_crds(i, j) = src_view(n_verts + face_leaf_idx(i), j);
        }
      });
}

template <typename SeedType>
void ScatterMeshData<SeedType>::scatter_phys_crds() {
  const auto face_mask = mesh.faces.mask;
  const auto face_leaf_idx = mesh.faces.leaf_idx;
  const Index n_verts = mesh.n_vertices_host();
  const Index n_faces = mesh.n_faces_host();

  const auto src_view = output.phys_crds;
  auto v_phys_crds = mesh.vertices.phys_crds.view;
  const auto vert_policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
      {0, 0}, {n_verts, SeedType::geo::ndim});

  Kokkos::parallel_for(
      "scatter_vert_phys_crds", vert_policy,
      KOKKOS_LAMBDA(const Index i, const int j) {
        v_phys_crds(i, j) = src_view(i, j);
      });

  const auto face_policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
      {0, 0}, {n_faces, SeedType::geo::ndim});
  auto f_phys_crds = mesh.faces.phys_crds.view;
  Kokkos::parallel_for(
      "scatter_face_phys_crds", face_policy,
      KOKKOS_LAMBDA(const Index i, const Int j) {
        if (!face_mask(i)) {
          f_phys_crds(i, j) = src_view(n_verts + face_leaf_idx(i), j);
        }
      });
}

template <typename SeedType>
void ScatterMeshData<SeedType>::scatter_fields(
    const vert_scalar_map& vertex_scalar_fields,
    const face_scalar_map& face_scalar_fields,
    const vert_vector_map& vertex_vector_fields,
    const face_vector_map& face_vector_fields) {
  const auto face_mask = mesh.faces.mask;
  const auto face_leaf_idx = mesh.faces.leaf_idx;
  const Index n_verts = mesh.n_vertices_host();
  const Index n_faces = mesh.n_faces_host();

  for (const auto& sf : vertex_scalar_fields) {
    LPM_REQUIRE_MSG(
        face_scalar_fields.count(sf.first) == 1,
        "ScatterMeshData::scatter_fields error: scalar output field " +
            sf.first + " not found.");
    if (output.scalar_fields.find(sf.first) != output.scalar_fields.end()) {
      auto mesh_vert_view = sf.second.view;
      auto mesh_face_view = face_scalar_fields.at(sf.first).view;
      const auto src_view = output.scalar_fields.at(sf.first);
      Kokkos::parallel_for(
          n_verts,
          KOKKOS_LAMBDA(const Index i) { mesh_vert_view(i) = src_view(i); });
      Kokkos::parallel_for(
          n_faces, KOKKOS_LAMBDA(const Index i) {
            if (!face_mask(i)) {
              mesh_face_view(i) = src_view(n_verts + face_leaf_idx(i));
            }
          });
    }
  }

  for (const auto& vf : vertex_vector_fields) {
    LPM_REQUIRE_MSG(
        face_vector_fields.count(vf.first) == 1,
        "ScatterMeshData::scatter_fields error: vector output field " +
            vf.first + " not found.");
    if (output.vector_fields.find(vf.first) != output.vector_fields.end()) {
      auto mesh_vert_view = vf.second.view;
      auto mesh_face_view = face_vector_fields.at(vf.first).view;
      const auto src_view = output.vector_fields.at(vf.first);
      Kokkos::parallel_for(
          n_verts, KOKKOS_LAMBDA(const Index i) {
            for (Int j = 0; j < SeedType::geo::ndim; ++j) {
              mesh_vert_view(i, j) = src_view(i, j);
            }
          });
      Kokkos::parallel_for(
          n_faces, KOKKOS_LAMBDA(const Index i) {
            if (!face_mask(i)) {
              for (Int j = 0; j < SeedType::geo::ndim; ++j) {
                mesh_face_view(i, j) = src_view(n_verts + face_leaf_idx(i), j);
              }
            }
          });
    }
  }
}

}  // namespace Lpm

#endif
