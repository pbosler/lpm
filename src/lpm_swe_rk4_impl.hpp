#ifndef LPM_SWE_RK4_IMPL_HPP
#define LPM_SWE_RK4_IMPL_HPP

#include "LpmConfig.h"

#include "lpm_swe_rk4.hpp"
#include "lpm_swe_kernels.hpp"

#include <KokkosBlas.hpp>

namespace Lpm {

template <typename Geo>
struct SWERK4Update {
  using crd_view = typename Geo::crd_view_type;

  static constexpr Real third = 1.0/3.0;
  static constexpr Real sixth = 1.0/6.0;

  crd_view x_inout;
  crd_view x1;
  crd_view x2;
  crd_view x3;
  crd_view x4;

  scalar_view_type zeta_inout;
  scalar_view_type zeta1;
  scalar_view_type zeta2;
  scalar_view_type zeta3;
  scalar_view_type zeta4;

  scalar_view_type sigma_inout;
  scalar_view_type sigma1;
  scalar_view_type sigma2;
  scalar_view_type sigma3;
  scalar_view_type sigma4;

  scalar_view_type h_or_a_inout;
  scalar_view_type h_or_a1;
  scalar_view_type h_or_a2;
  scalar_view_type h_or_a3;
  scalar_view_type h_or_a4;

  SWERK4Update(crd_view& x_inout,
    const crd_view x1, const crd_view x2, const crd_view x3, const crd_view x4,
    scalar_view_type& zeta_inout,
    const scalar_view_type zeta1, const scalar_view_type zeta2,
    const scalar_view_type zeta3, const scalar_view_type zeta4,
    scalar_view_type& sigma_inout,
    const scalar_view_type sigma1, const scalar_view_type sigma2,
    const scalar_view_type sigma3, const scalar_view_type sigma4,
    scalar_view_type& h_or_a_inout,
    const scalar_view_type h_or_a1, const scalar_view_type h_or_a2,
    const scalar_view_type h_or_a3, const scalar_view_type h_or_a4) :
    x_inout(x_inout),
    x1(x1),
    x2(x2),
    x3(x3),
    x4(x4),
    zeta_inout(zeta_inout),
    zeta1(zeta1),
    zeta2(zeta2),
    zeta3(zeta3),
    zeta4(zeta4),
    sigma_inout(sigma_inout),
    sigma1(sigma1),
    sigma2(sigma2),
    sigma3(sigma3),
    sigma4(sigma4),
    h_or_a_inout(h_or_a_inout),
    h_or_a1(h_or_a1),
    h_or_a2(h_or_a2),
    h_or_a3(h_or_a3),
    h_or_a4(h_or_a4) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index i) const {
    for (int j=0; j<Geo::ndim; ++j) {
      x_inout(i,j) += sixth * (x1(i,j) + x4(i,j)) + third * (x2(i,j) + x3(i,j));
    }
    zeta_inout(i) += sixth * (zeta1(i) + zeta4(i)) + third * (zeta2(i) + zeta3(i));
    sigma_inout(i) += sixth * (sigma1(i) + sigma4(i)) + third * (sigma2(i) + sigma3(i));
    h_or_a_inout(i) += sixth * (h_or_a1(i) + h_or_a4(i)) + third * (h_or_a2(i) + h_or_a3(i));
  }
};

// constructor
template <typename SeedType, typename TopoType>
SWERK4<SeedType, TopoType>::SWERK4(const Real timestep, SWE<SeedType>& swe, TopoType& topo):
  dt(timestep),
  t_idx(0),
  swe(swe),
  topo(topo),
  eps(swe.eps),
  pse_eps(swe.pse_eps),
  passive_x1("passive_x1", swe.mesh.n_vertices_host()),
  passive_x2("passive_x2", swe.mesh.n_vertices_host()),
  passive_x3("passive_x3", swe.mesh.n_vertices_host()),
  passive_x4("passive_x4", swe.mesh.n_vertices_host()),
  passive_xwork("passive_xwork", swe.mesh.n_vertices_host()),
  passive_rel_vort1("passive_rel_vort1", swe.mesh.n_vertices_host()),
  passive_rel_vort2("passive_rel_vort2", swe.mesh.n_vertices_host()),
  passive_rel_vort3("passive_rel_vort3", swe.mesh.n_vertices_host()),
  passive_rel_vort4("passive_rel_vort4", swe.mesh.n_vertices_host()),
  passive_rel_vortwork("passive_rel_vortwork", swe.mesh.n_vertices_host()),
  passive_div1("passive_div1", swe.mesh.n_vertices_host()),
  passive_div2("passive_div2", swe.mesh.n_vertices_host()),
  passive_div3("passive_div3", swe.mesh.n_vertices_host()),
  passive_div4("passive_div4", swe.mesh.n_vertices_host()),
  passive_divwork("passive_divwork", swe.mesh.n_vertices_host()),
  passive_depth1("passive_depth1", swe.mesh.n_vertices_host()),
  passive_depth2("passive_depth2", swe.mesh.n_vertices_host()),
  passive_depth3("passive_depth3", swe.mesh.n_vertices_host()),
  passive_depth4("passive_depth4", swe.mesh.n_vertices_host()),
  passive_depthwork("passive_depthwork", swe.mesh.n_vertices_host()),
  active_x1("active_x1", swe.mesh.n_faces_host()),
  active_x2("active_x2", swe.mesh.n_faces_host()),
  active_x3("active_x3", swe.mesh.n_faces_host()),
  active_x4("active_x4", swe.mesh.n_faces_host()),
  active_xwork("active_xwork", swe.mesh.n_faces_host()),
  active_rel_vort1("active_rel_vort1", swe.mesh.n_faces_host()),
  active_rel_vort2("active_rel_vort2", swe.mesh.n_faces_host()),
  active_rel_vort3("active_rel_vort3", swe.mesh.n_faces_host()),
  active_rel_vort4("active_rel_vort4", swe.mesh.n_faces_host()),
  active_rel_vortwork("active_rel_vortwork", swe.mesh.n_faces_host()),
  active_div1("active_div1", swe.mesh.n_faces_host()),
  active_div2("active_div2", swe.mesh.n_faces_host()),
  active_div3("active_div3", swe.mesh.n_faces_host()),
  active_div4("active_div4", swe.mesh.n_faces_host()),
  active_divwork("active_divwork", swe.mesh.n_faces_host()),
  active_area1("active_area1", swe.mesh.n_faces_host()),
  active_area2("active_area2", swe.mesh.n_faces_host()),
  active_area3("active_area3", swe.mesh.n_faces_host()),
  active_area4("active_area4", swe.mesh.n_faces_host()),
  active_areawork("active_areawork", swe.mesh.n_faces_host()) {
    passive_policy = std::make_unique<Kokkos::TeamPolicy<>>(swe.mesh.n_vertices_host(), Kokkos::AUTO());
    active_policy = std::make_unique<Kokkos::TeamPolicy<>>(swe.mesh.n_faces_host(), Kokkos::AUTO());
    set_fixed_views();
  }

template <typename SeedType, typename TopoType>
void SWERK4<SeedType, TopoType>::set_fixed_views() {
  passive_x = swe.mesh.vertices.phys_crds.view;
  passive_rel_vort = swe.rel_vort_passive.view;
  passive_divergence = swe.div_passive.view;
  passive_depth = swe.depth_passive.view;
  passive_surface = swe.surf_passive.view;
  passive_vel = swe.velocity_passive.view;
  passive_ddot = swe.double_dot_passive.view;
  passive_laps = swe.surf_lap_passive.view;
  passive_bottom = swe.bottom_passive.view;

  active_x = swe.mesh.faces.phys_crds.view;
  active_rel_vort = swe.rel_vort_active.view;
  active_divergence = swe.div_active.view;
  active_depth = swe.depth_active.view;
  active_surface = swe.surf_active.view;
  active_vel = swe.velocity_active.view;
  active_ddot = swe.double_dot_active.view;
  active_laps = swe.surf_lap_active.view;
  active_bottom = swe.bottom_active.view;
  active_mass = swe.mass_active.view;
  active_area = swe.mesh.faces.area;
  active_mask = swe.mesh.faces.mask;
}

template <typename SeedType, typename TopoType>
std::string SWERK4<SeedType, TopoType>::info_string(const int tab_level) const {
  std::ostringstream ss;
  ss << "SWERK4 info:\n"
     << "dt = " << dt << " t_idx = " << t_idx << "\n"
     << "eps = " << eps << " pse_eps = " << pse_eps << "\n";
  return ss.str();
}

template <typename SeedType, typename TopoType>
void SWERK4<SeedType, TopoType>::advance_timestep() {
  return advance_timestep(swe.mesh.vertices.phys_crds.view,
    swe.rel_vort_passive.view,
    swe.div_passive.view,
    swe.depth_passive.view,
    swe.velocity_passive.view,
    swe.mesh.faces.phys_crds.view,
    swe.rel_vort_active.view,
    swe.div_active.view,
    swe.depth_active.view,
    swe.velocity_active.view,
    swe.mesh.faces.area,
    swe.mass_active.view,
    swe.mesh.faces.mask);
}

template <typename SeedType, typename TopoType>
void SWERK4<SeedType, TopoType>::advance_timestep(crd_view& vx,
                                        scalar_view_type& vzeta,
                                        scalar_view_type& vsigma,
                                        scalar_view_type& vh,
                                        vec_view& vvel,
                                        crd_view& fx,
                                        scalar_view_type& fzeta,
                                        scalar_view_type& fsigma,
                                        scalar_view_type& fh,
                                        vec_view& fvel,
                                        scalar_view_type& farea,
                                        const scalar_view_type& fmass,
                                        const mask_view_type& fmask)
{

  set_fixed_views();
  const Index nverts = swe.mesh.n_vertices_host();
  const Index nfaces = swe.mesh.n_faces_host();
  constexpr bool do_velocity = true;

  /// RK Stage 1
  /// velocity is already defined on the mesh; scale by dt
  KokkosBlas::scal(passive_x1, dt, passive_vel);
  KokkosBlas::scal(active_x1, dt, active_vel);
  /// compute tendencies for zeta, sigma, h (passive particles)
  Kokkos::parallel_for("RK4-1 vertex tendencies",
    nverts,
    PlanarSWEVorticityDivergenceHeightTendencies(
      passive_rel_vort1, passive_div1, passive_depth1,
      passive_x,
      passive_rel_vort, passive_divergence, passive_depth,
      passive_ddot, passive_laps, swe.coriolis, swe.g, dt));
  /// compute tendencies for zeta, sigma, area (active particles)
  Kokkos::parallel_for("RK4-1 face tendencies",
    nfaces,
    PlanarSWEVorticityDivergenceAreaTendencies(
      active_rel_vort1, active_div1, active_area1,
      active_x,
      active_rel_vort, active_divergence, active_area,
      active_ddot, active_laps, swe.coriolis, swe.g, dt));

  /// RK Stage 2
  /// update input positions to time t + 0.5*dt
  KokkosBlas::update(1.0, passive_x, 0.5, passive_x1, 0.0, passive_xwork);
  KokkosBlas::update(1.0, active_x, 0.5, active_x1, 0.0, active_xwork);;
  /// update input vorticity to time t + 0.5*dt
  KokkosBlas::update(1.0, passive_rel_vort, 0.5, passive_rel_vort1, 0, passive_rel_vortwork);
  KokkosBlas::update(1.0, active_rel_vort, 0.5, active_rel_vort1, 0, active_rel_vortwork);
  /// update input divergence to time t + 0.5*dt
  KokkosBlas::update(1.0, passive_divergence, 0.5, passive_div1, 0, passive_divwork);
  KokkosBlas::update(1.0, active_divergence, 0.5, active_div1, 0, active_divwork);
  /// update input depth to time t + 0.5*dt
  KokkosBlas::update(1.0, passive_depth, 0.5, passive_depth1, 0, passive_depthwork);
  /// update input area to time t + 0.5*dt
  KokkosBlas::update(1.0, active_area, 0.5, active_area1, 0, active_areawork);
  /// update input surface to time t + 0.5*dt
  Kokkos::parallel_for("RK4-2 surface input, passive",
    nverts,
    SetSurfaceFromDepth<typename SeedType::geo, TopoType>(passive_surface, passive_bottom, passive_xwork, passive_depthwork, topo));
  Kokkos::parallel_for("RK4-2, surface input, active",
    nfaces,
    SetDepthAndSurfaceFromMassAndArea<typename SeedType::geo, TopoType>(active_depth, active_surface, active_bottom,
      active_xwork, active_mass, active_areawork, active_mask, topo));
  /// compute velocity, double dot, and surface laplacian
  Kokkos::parallel_for("RK4-2 direct sum, passive",
    *passive_policy,
    PlanarSWEVertexSums(passive_vel, passive_ddot, passive_laps,
      passive_xwork, passive_surface, active_xwork, active_rel_vortwork,
      active_divwork, active_areawork, active_mask, active_surface,
      eps, pse_eps, nfaces, do_velocity));
  Kokkos::parallel_for("RK4-2 direct sum, active",
    *active_policy,
    PlanarSWEFaceSums(active_vel, active_ddot, active_laps,
      active_xwork, active_rel_vortwork, active_divwork, active_areawork,
      active_mask, active_surface, eps, pse_eps, nfaces, do_velocity));
  /// compute vorticity, divergence, and height tendencies
  Kokkos::parallel_for("RK4-2 passive tendencies",
    nverts,
    PlanarSWEVorticityDivergenceHeightTendencies(
      passive_rel_vort2, passive_div2, passive_depth2,
      passive_xwork,
      passive_rel_vortwork, passive_divwork, passive_depthwork, passive_ddot, passive_laps,
      swe.coriolis, swe.g, dt));
  Kokkos::parallel_for("RK4-2 active tendencies",
    nfaces,
    PlanarSWEVorticityDivergenceAreaTendencies(
      active_rel_vort2, active_div2, active_area2,
      active_xwork,
      active_rel_vortwork, active_divwork, active_areawork,
      active_ddot, active_laps, swe.coriolis, swe.g, dt));

  /// RK Stage 3
  /// second update input to t + 0.5 * dt
  KokkosBlas::scal(passive_x2, dt, passive_vel);
  KokkosBlas::update(1.0, passive_x, 0.5, passive_x2, 0, passive_xwork);
  KokkosBlas::scal(active_x2, dt, active_vel);
  KokkosBlas::update(1.0, active_x, 0.5, active_x2, 0, active_xwork);
  KokkosBlas::update(1.0, passive_rel_vort, 0.5, passive_rel_vort2, 0, passive_rel_vortwork);
  KokkosBlas::update(1.0, active_rel_vort, 0.5, active_rel_vort2, 0, active_rel_vortwork);
  KokkosBlas::update(1.0, passive_divergence, 0.5, passive_div2, 0, passive_divwork);
  KokkosBlas::update(1.0, active_divergence, 0.5, active_div2, 0, active_divwork);
  KokkosBlas::update(1.0, passive_depth, 0.5, passive_depth2, 0, passive_depthwork);
  KokkosBlas::update(1.0, active_area, 0.5, active_area2, 0, active_areawork);
  Kokkos::parallel_for("RK4-3 surface input, passive",
    nverts,
    SetSurfaceFromDepth<typename SeedType::geo, TopoType>(passive_surface, passive_bottom, passive_xwork, passive_depthwork, topo));
  Kokkos::parallel_for("RK4-3, surface input, active",
    nfaces,
    SetDepthAndSurfaceFromMassAndArea<typename SeedType::geo, TopoType>(active_depth, active_surface, active_bottom,
      active_xwork, active_mass, active_areawork, active_mask, topo));
  Kokkos::parallel_for("RK4-3 direct sum, passive",
    *passive_policy,
    PlanarSWEVertexSums(passive_vel, passive_ddot, passive_laps,
      passive_xwork, passive_surface, active_xwork, active_rel_vortwork,
      active_divwork, active_areawork, active_mask, active_surface,
      eps, pse_eps, nfaces, do_velocity));
  Kokkos::parallel_for("RK4-3 direct sum, active",
    *active_policy,
    PlanarSWEFaceSums(active_vel, active_ddot, active_laps,
      active_xwork, active_rel_vortwork, active_divwork, active_areawork,
      active_mask, active_surface, eps, pse_eps, nfaces, do_velocity));
  Kokkos::parallel_for("RK4-3 passive tendencies",
    nverts,
    PlanarSWEVorticityDivergenceHeightTendencies(
      passive_rel_vort3, passive_div3, passive_depth3,
      passive_xwork,
      passive_rel_vortwork, passive_divwork, passive_depthwork, passive_ddot, passive_laps,
      swe.coriolis, swe.g, dt));
  Kokkos::parallel_for("RK4-3 active tendencies",
    nfaces,
    PlanarSWEVorticityDivergenceAreaTendencies(
      active_rel_vort3, active_div3, active_area3,
      active_xwork,
      active_rel_vortwork, active_divwork, active_areawork,
      active_ddot, active_laps, swe.coriolis, swe.g, dt));

  /// RK Stage 4
  /// update input to t + dt
  KokkosBlas::scal(passive_x3, dt, passive_vel);
  KokkosBlas::scal(active_x3, dt, active_vel);
  KokkosBlas::update(1.0, passive_x, 1, passive_x3, 0, passive_xwork);
  KokkosBlas::update(1.0, active_x, 1, active_x3, 0, active_xwork);
  KokkosBlas::update(1.0, passive_rel_vort, 1.0, passive_rel_vort3, 0, passive_rel_vortwork);
  KokkosBlas::update(1.0, active_rel_vort, 1.0, active_rel_vort3, 0, active_rel_vortwork);
  KokkosBlas::update(1.0, passive_divergence, 1.0, passive_div3, 0, passive_divwork);
  KokkosBlas::update(1.0, active_divergence, 1.0, active_div3, 0, active_divwork);
  KokkosBlas::update(1.0, passive_depth, 1.0, passive_depth3, 0, passive_depthwork);
  KokkosBlas::update(1.0, active_area, 1.0, active_area3, 0, active_areawork);
  Kokkos::parallel_for("RK4-4 surface input, passive",
    nverts,
    SetSurfaceFromDepth<typename SeedType::geo, TopoType>(passive_surface, passive_bottom, passive_xwork, passive_depthwork, topo));
  Kokkos::parallel_for("RK4-4, surface input, active",
    nfaces,
    SetDepthAndSurfaceFromMassAndArea<typename SeedType::geo, TopoType>(active_depth, active_surface, active_bottom,
      active_xwork, active_mass, active_areawork, active_mask, topo));
  Kokkos::parallel_for("RK4-4 direct sum, passive",
    *passive_policy,
    PlanarSWEVertexSums(passive_vel, passive_ddot, passive_laps,
      passive_xwork, passive_surface, active_xwork, active_rel_vortwork,
      active_divwork, active_areawork, active_mask, active_surface,
      eps, pse_eps, nfaces, do_velocity));
  Kokkos::parallel_for("RK4-4 direct sum, active",
    *active_policy,
    PlanarSWEFaceSums(active_vel, active_ddot, active_laps,
      active_xwork, active_rel_vortwork, active_divwork, active_areawork,
      active_mask, active_surface, eps, pse_eps, nfaces, do_velocity));
  Kokkos::parallel_for("RK4-4 passive tendencies",
    nverts,
    PlanarSWEVorticityDivergenceHeightTendencies(
      passive_rel_vort4, passive_div4, passive_depth4, passive_xwork,
      passive_rel_vortwork, passive_divwork, passive_depthwork, passive_ddot, passive_laps,
      swe.coriolis, swe.g, dt));
  Kokkos::parallel_for("RK4-4 active tendencies",
    nfaces,
    PlanarSWEVorticityDivergenceAreaTendencies(
      active_rel_vort4, active_div4, active_area4, active_xwork,
      active_rel_vortwork, active_divwork, active_areawork,
      active_ddot, active_laps, swe.coriolis, swe.g, dt));


  /// RK Final: set time step output
  Kokkos::parallel_for("RK4 output, passive",
    nverts,
    SWERK4Update<PlaneGeometry>(passive_x, passive_x1, passive_x2, passive_x3, passive_x4,
      passive_rel_vort, passive_rel_vort1, passive_rel_vort2, passive_rel_vort3, passive_rel_vort4,
      passive_divergence, passive_div1, passive_div2, passive_div3, passive_div4,
      passive_depth, passive_depth1, passive_depth2, passive_depth3, passive_depth4));
  Kokkos::parallel_for("RK4 output, active",
    nfaces,
    SWERK4Update<PlaneGeometry>(active_x, active_x1, active_x2, active_x3, active_x4,
      active_rel_vort, active_rel_vort1, active_rel_vort2, active_rel_vort3, active_rel_vort4,
      active_divergence, active_div1, active_div2, active_div3, active_div4,
      active_area, active_area1, active_area2, active_area3, active_area4));
  Kokkos::parallel_for("RK4-final surface, passive",
    nverts,
    SetSurfaceFromDepth<typename SeedType::geo, TopoType>(passive_surface, passive_bottom, passive_x, passive_depth, topo));
  Kokkos::parallel_for("RK4-4-final, surface, active",
    nfaces,
    SetDepthAndSurfaceFromMassAndArea<typename SeedType::geo, TopoType>(active_depth, active_surface, active_bottom,
      active_x, active_mass, active_area, active_mask, topo));
  Kokkos::parallel_for("RK4-final direct sum, passive",
    *passive_policy,
    PlanarSWEVertexSums(passive_vel, passive_ddot, passive_laps,
      passive_x, passive_surface, active_x, active_rel_vort,
      active_divergence, active_area, active_mask, active_surface,
      eps, pse_eps, nfaces, do_velocity));
  Kokkos::parallel_for("RK4-final direct sum, active",
    *active_policy,
    PlanarSWEFaceSums(active_vel, active_ddot, active_laps,
      active_x, active_rel_vort, active_divergence, active_area,
      active_mask, active_surface, eps, pse_eps, nfaces, do_velocity));

  ++t_idx;
}



} // namespace Lpm

#endif
