#ifndef LPM_SWE_RK2_IMPL_HPP
#define LPM_SWE_RK2_IMPL_HPP

#include "LpmConfig.h"
#include "lpm_swe_rk2.hpp"
#include "lpm_swe_kernels.hpp"
#include "mesh/lpm_gather_mesh_data_impl.hpp"
#include "mesh/lpm_scatter_mesh_data_impl.hpp"

#include <KokkosBlas.hpp>

namespace Lpm {

template <typename SeedType, typename TopoType>
SWERK2<SeedType,TopoType>::SWERK2(const Real timestep, SWE<SeedType>& swe, TopoType& topo, const gmls::Params& gmls_params) :
  dt(timestep),
  t_idx(0),
  swe(swe),
  topo(topo),
  eps(swe.eps),
  gmls_params(gmls_params),
  passive_x1("passive_x1", swe.mesh.n_vertices_host()),
  passive_x2("passive_x2", swe.mesh.n_vertices_host()),
  passive_xwork("passive_xwork", swe.mesh.n_vertices_host()),
  passive_rel_vort1("passive_rel_vort1", swe.mesh.n_vertices_host()),
  passive_rel_vort2("passive_rel_vort2", swe.mesh.n_vertices_host()),
  passive_rel_vortwork("passive_rel_vortwork", swe.mesh.n_vertices_host()),
  passive_div1("passive_div1", swe.mesh.n_vertices_host()),
  passive_div2("passive_div2", swe.mesh.n_vertices_host()),
  passive_divwork("passive_divwork", swe.mesh.n_vertices_host()),
  passive_depth1("passive_depth1", swe.mesh.n_vertices_host()),
  passive_depth2("passive_depth2", swe.mesh.n_vertices_host()),
  passive_depthwork("passive_depthwork", swe.mesh.n_vertices_host()),
  active_x1("active_x1", swe.mesh.n_vertices_host()),
  active_x2("active_x2", swe.mesh.n_vertices_host()),
  active_xwork("active_xwork", swe.mesh.n_vertices_host()),
  active_rel_vort1("active_rel_vort1", swe.mesh.n_vertices_host()),
  active_rel_vort2("active_rel_vort2", swe.mesh.n_vertices_host()),
  active_rel_vortwork("active_rel_vortwork", swe.mesh.n_vertices_host()),
  active_div1("active_div1", swe.mesh.n_vertices_host()),
  active_div2("active_div2", swe.mesh.n_vertices_host()),
  active_divwork("active_divwork", swe.mesh.n_vertices_host()),
  active_area1("active_area1", swe.mesh.n_vertices_host()),
  active_area2("active_area2", swe.mesh.n_vertices_host()),
  active_areawork("active_areawork", swe.mesh.n_vertices_host())
{
  passive_policy = std::make_unique<Kokkos::TeamPolicy<>>(swe.mesh.n_vertices_host(), Kokkos::AUTO());
  active_policy = std::make_unique<Kokkos::TeamPolicy<>>(swe.mesh.n_faces_host(), Kokkos::AUTO());

  gather = std::make_unique<GatherMeshData<SeedType>>(swe.mesh);
  scatter = std::make_unique<ScatterMeshData<SeedType>>(*gather, swe.mesh);

  passive_field_map.emplace("surface_height", swe.surf_passive);
  passive_field_map.emplace("surface_laplacian", swe.surf_lap_passive);
  active_field_map.emplace("surface_height", swe.surf_active);
  active_field_map.emplace("surface_laplacian", swe.surf_lap_active);

  gather->init_scalar_fields(passive_field_map, active_field_map);
  gather->gather_scalar_fields(passive_field_map, active_field_map);
  const auto neighbors = gmls::Neighborhoods(gather->h_phys_crds,
    gmls_params);
  gmls_ops = std::vector<Compadre::TargetOperation>(
    {Compadre::LaplacianOfScalarPointEvaluation});
  auto surf_gmls = gmls::sphere_scalar_gmls(gather->phys_crds, gather->phys_crds,
    neighbors, gmls_params, gmls_ops);
  auto eval = Compadre::Evaluator(&surf_gmls);
  gather->scalar_fields.at("surface_laplacian") =
    eval.applyAlphasToDataAllComponentsAllTargetSites<Real*,DevMemory>(
      gather->scalar_fields.at("surface_height"),
      Compadre::LaplacianOfScalarPointEvaluation,
      Compadre::PointSample);
  scatter->scatter_fields(passive_field_map, active_field_map);
}

template <typename SeedType, typename TopoType>
void SWERK2<SeedType,TopoType>::advance_timestep_impl() {
  const Index nverts = swe.mesh.n_vertices_host();
  const Index nfaces = swe.mesh.n_faces_host();
  constexpr bool do_velocity = true;
  /// RK Stage 1
  KokkosBlas::scal(passive_x1, dt, swe.velocity_passive.view);
  KokkosBlas::scal(active_x1, dt, swe.velocity_active.view);

  Kokkos::parallel_for("RK2-1 vertex tendencies",
    nverts,
    SWEVorticityDivergenceHeightTendencies<geo>(
      passive_rel_vort1, passive_div1, passive_depth1,
      swe.mesh.vertices.phys_crds.view,
      swe.rel_vort_passive.view, swe.div_passive.view, swe.depth_passive.view,
      swe.double_dot_passive.view, swe.surf_lap_passive.view, swe.coriolis, swe.g, dt));

  Kokkos::parallel_for("RK2-1 face tendencies",
    nfaces,
    SWEVorticityDivergenceAreaTendencies<geo>(active_rel_vort1,
      active_div1, active_area1, swe.mesh.faces.phys_crds.view,
      swe.rel_vort_active.view, swe.div_active.view, swe.mesh.faces.area,
      swe.double_dot_active.view, swe.surf_lap_active.view, swe.coriolis, swe.g, dt));

  /// RK Stage 2
  /// intermediate update to time t + dt
  KokkosBlas::update(1, swe.mesh.vertices.phys_crds.view, 1, passive_x1, 0, passive_xwork);
  KokkosBlas::update(1, swe.rel_vort_passive.view, 1, passive_rel_vort1, 0, passive_rel_vortwork);
  KokkosBlas::update(1, swe.div_passive.view, 1, passive_div1, 0, passive_divwork);
  KokkosBlas::update(1, swe.mesh.faces.phys_crds.view, 1, active_x1, 0, active_xwork);
  KokkosBlas::update(1, swe.rel_vort_active.view, 1, active_rel_vort1, 0, active_rel_vortwork);
  KokkosBlas::update(1, swe.div_active.view, 1, active_div1, 0, active_divwork);
  KokkosBlas::update(1, swe.depth_passive.view, 1, passive_depth1, 0, passive_depthwork);
  KokkosBlas::udpate(1, swe.mesh.faces.area, 1, active_area1, 0, active_areawork);
  Kokkos::parallel_for("RK2-2 surface input, passive",
    nverts,
    SetSurfaceFromDepth<geo, TopoType>(swe.surf_passive.view, swe.bottom_passive.view,
      passive_xwork, passive_depthwork, topo));
  Kokkos::parallel_for("RK2-2 surface input, active",
    nfaces,
    SetDepthAndSurfaceFromMassAndArea(swe.depth_active.view, swe.surf_active.view,
      swe.bottom_active.view, active_xwork, swe.mass_active.view,
      active_areawork, swe.mesh.faces.mask, topo));
  if constexpr (std::is_same<geo, SphereGeometry>::value) {
    Kokkos::parallel_for("RK2-2 direct sums, passive",
      *passive_policy,
      SphereVertexSums(swe.velocity_passive.view, swe.double_dot_passive.view,
          passive_xwork, active_xwork, active_rel_vortwork,
          active_divwork, active_areawork, swe.mesh.faces.mask, eps, nfaces,
          do_velocity));
    Kokkos::parallel_for("RK2-2 direct sums, active",
      *active_policy,
      SphereFaceSums(swe.velocity_active.view, swe.double_dot_active.view,
        active_xwork, active_rel_vortwork, active_divwork, active_areawork,
        swe.mesh.faces.mask, eps, nfaces, do_velocity));
    /// update surface Laplacian
    gather->gather_coordindates(passive_xwork, active_xwork);
    gather->update_host();
    {
      const auto neighbors = gmls::Neighborhoods(gather->h_phys_crds, gmls_params);
      auto surf_gmls = gmls::sphere_scalar_gmls(gather->phys_crds, gather->phys_crds,
        neighbors, gmls_params, gmls_ops);
      auto eval = Compadre::Evaluator(&surf_gmls);
      gather->scalar_fields.at("surface_laplacian") =
        eval.applyAlphasToDataAllComponentsAllTargetSites<Real*,DevMemory>(
        gather->scalar_fields.at("surface_height"),
        Compadre::LaplacianOfScalarPointEvaluation, Compadre::PointSample);
      scatter->scatter_fields(passive_field_map, active_field_map);
    }
  }
  else {
    // TODO : planar RK2 goes here
  }
}

template <typename SeedType, typename TopoType>
std::string SWERK2<SeedType,TopoType>::info_string(const int tab_level) const {
  std::ostringstream ss;
  return ss.str();
}

} // namespace Lpm

#endif
