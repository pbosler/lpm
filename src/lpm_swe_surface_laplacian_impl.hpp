#ifndef LPM_SWE_SURFACE_LAPLACIAN_IMPL_HPP
#define LPM_SWE_SURFACE_LAPLACIAN_IMPL_HPP

#include "lpm_swe_surface_laplacian.hpp"
#include "mesh/lpm_gather_mesh_data_impl.hpp"
#include "mesh/lpm_scatter_mesh_data_impl.hpp"

namespace Lpm {

template <typename SeedType>
SWEPSELaplacian<SeedType>::SWEPSELaplacian(SWE<SeedType> &swe,
                                           const Real pse_eps)
    : surf_lap_passive(
          Kokkos::subview(swe.surf_laplacian_passive.view,
                          std::make_pair(0, swe.mesh.n_vertices_host()))),
      surf_lap_active(
          Kokkos::subview(swe.surf_laplacian_active.view,
                          std::make_pair(0, swe.mesh.n_faces_host()))),
      x_passive(Kokkos::subview(swe.mesh.vertices.phys_crds.view,
                                std::make_pair(0, swe.mesh.n_vertices_host()),
                                Kokkos::ALL)),
      x_active(Kokkos::subview(swe.mesh.faces.phys_crds.view,
                               std::make_pair(0, swe.mesh.n_faces_host()),
                               Kokkos::ALL)),
      surface_passive(
          Kokkos::subview(swe.surf_passive.view,
                          std::make_pair(0, swe.mesh.n_vertices_host()))),
      surface_active(Kokkos::subview(
          swe.surf_active.view, std::make_pair(0, swe.mesh.n_faces_host()))),
      area_active(Kokkos::subview(swe.mesh.faces.area,
                                  std::make_pair(0, swe.mesh.n_faces_host()))),
      n_passive(swe.mesh.n_vertices_host()), n_active(swe.mesh.n_faces_host()),
      eps(pse_eps) {}

template <typename SeedType>
void SWEPSELaplacian<SeedType>::update_src_data(const crd_view xp,
                                                const crd_view xa,
                                                const scalar_view_type sp,
                                                const scalar_view_type sa,
                                                const scalar_view_type ar) {
  x_passive = xp;
  x_active = xa;
  surface_passive = sp;
  surface_active = sa;
  area_active = ar;
}

template <typename SeedType> void SWEPSELaplacian<SeedType>::compute() {
  Kokkos::TeamPolicy<> passive_policy(n_passive, Kokkos::AUTO());
  Kokkos::TeamPolicy<> active_policy(n_active, Kokkos::AUTO());

  Kokkos::parallel_for(
      "PSE Laplacian (passive)", passive_policy,
      pse::ScalarLaplacian<pse_type>(surf_lap_passive, x_passive, x_active,
                                     surface_passive, surface_active,
                                     area_active, eps, n_active));

  Kokkos::parallel_for("PSE Laplacian (active)", active_policy,
                       pse::ScalarLaplacian<pse_type>(
                           surf_lap_active, x_active, x_active, surface_active,
                           surface_active, area_active, eps, n_active));
}

template <typename SeedType>
SWEGMLSLaplacian<SeedType>::SWEGMLSLaplacian(SWE<SeedType> &swe,
                                             const gmls::Params &params)
    : surf_lap_passive(
          Kokkos::subview(swe.surf_laplacian_passive.view,
                          std::make_pair(0, swe.mesh.n_vertices_host()))),
      surf_lap_active(
          Kokkos::subview(swe.surf_laplacian_active.view,
                          std::make_pair(0, swe.mesh.n_faces_host()))),
      x_passive(Kokkos::subview(swe.mesh.vertices.phys_crds.view,
                                std::make_pair(0, swe.mesh.n_vertices_host()),
                                Kokkos::ALL)),
      x_active(Kokkos::subview(swe.mesh.faces.phys_crds.view,
                               std::make_pair(0, swe.mesh.n_faces_host()),
                               Kokkos::ALL)),
      surface_passive(
          Kokkos::subview(swe.surf_passive.view,
                          std::make_pair(0, swe.mesh.n_vertices_host()))),
      surface_active(Kokkos::subview(
          swe.surf_active.view, std::make_pair(0, swe.mesh.n_faces_host()))),
      area_active(Kokkos::subview(swe.mesh.faces.area,
                                  std::make_pair(0, swe.mesh.n_faces_host()))),
      n_passive(swe.mesh.n_vertices_host()), n_active(swe.mesh.n_faces_host()),
      params(params) {

  gathered_mesh = std::make_unique<GatherMeshData<SeedType>>(swe.mesh);
  gathered_surface = scalar_view_type("surface_height", gathered_mesh->n());
  gathered_laplacian =
      scalar_view_type("surface_laplacian", gathered_mesh->n());
  gathered_mesh->init_scalar_field("surface_height", gathered_surface);
  gathered_mesh->init_scalar_field("surface_laplacian", gathered_laplacian);

  scatter_mesh =
      std::make_unique<ScatterMeshData<SeedType>>(*gathered_mesh, swe.mesh);
  neighbors = gmls::Neighborhoods(gathered_mesh->h_phys_crds,
                                  gathered_mesh->h_phys_crds, params);
}

template <typename SeedType>
void SWEGMLSLaplacian<SeedType>::update_src_data(const crd_view xp,
                                                 const crd_view xa,
                                                 const scalar_view_type sp,
                                                 const scalar_view_type sa,
                                                 const scalar_view_type ar) {
  x_passive = xp;
  x_active = xa;
  surface_passive = sp;
  surface_active = sa;
  area_active = ar;
  gathered_mesh->gather_phys_coordinates(x_passive, x_active);
  gathered_mesh->gather_scalar_field("surface_height", surface_passive,
                                     surface_active);
  neighbors.update_neighbors(gathered_mesh->h_phys_crds,
                             gathered_mesh->h_phys_crds, params);
  const std::vector<Compadre::TargetOperation> ops(
      {Compadre::LaplacianOfScalarPointEvaluation});
  if constexpr (std::is_same<geo, SphereGeometry>::value) {
    scalar_gmls = std::make_unique<Compadre::GMLS>(gmls::sphere_scalar_gmls(
        gathered_mesh->phys_crds, gathered_mesh->phys_crds, neighbors, params,
        ops));
  } else {
    scalar_gmls = std::make_unique<Compadre::GMLS>(gmls::plane_scalar_gmls(
        gathered_mesh->phys_crds, gathered_mesh->phys_crds, neighbors, params,
        ops));
  }
}

template <typename SeedType> void SWEGMLSLaplacian<SeedType>::compute() {
  Compadre::Evaluator gmls_eval_scalar(scalar_gmls.get());
  const auto gmls_laplacian =
      gmls_eval_scalar
          .applyAlphasToDataAllComponentsAllTargetSites<Real *, DevMemory>(
              gathered_surface, Compadre::LaplacianOfScalarPointEvaluation,
              Compadre::PointSample);
  Kokkos::deep_copy(gathered_laplacian, gmls_laplacian);
  scatter_mesh->scatter_scalar_field("surface_laplacian", surf_lap_passive,
                                     surf_lap_active);
}

} // namespace Lpm
#endif
