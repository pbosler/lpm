#ifndef LPM_HEADER_HPP
#define LPM_HEADER_HPP

#include "LpmConfig.h"
#include "lpm_swe_solver.hpp"
#include "lpm_swe_kernels.hpp"
#include "lpm_swe_tendencies.hpp"
#include "util/lpm_tuple.hpp"

#include <KokkosBlas.hpp>

namespace Lpm {

template <typename SeedType>
SWERK4<SeedType>::SWERK4(SWE<SeedType>& swe, const Real dt) :
  swe(swe),
  dt(dt),
  x_passive(
    Kokkos::subview(swe.mesh.vertices.phys_crds.view,
    std::make_pair(0, mesh.n_vertices_host()), Kokkos::ALL)),
  x_active(
    Kokkos::subview(swe.mesh.faces.phys_crds.view,
    std::make_pair(0, mesh.n_faces_host()), Kokkos::ALL)),
  velocity_passive(
    Kokkos::subview(swe.velocity_passive,
    std::make_pair(0, swe.mesh.n_vertices_host()), Kokkos::ALL)),
  velocity_active(
    Kokkos::subview(swe.velocity_active,
    std::make_pair(0, swe.mesh.n_faces_host()), Kokkos::ALL)),
  rel_vort_passive(
    Kokkos::subview(swe.rel_vort_passive, std::make_pair(0, swe.mesh.n_vertices_host()))),
  rel_vort_active(
    Kokkos::subview(swe.rel_vort_active.view, std::make_pair(0, mesh.n_faces_host()))),
  divergence_passive(
    Kokkos::subview(swe.divergence_passive.view, std::make_pair(0, swe.mesh.n_vertices_host()))),
  divergence_active(
    Kokkos::subview(swe.divergence_active.view, std::make_pair(0, swe.mesh.n_faces_host()))),
  depth_passive(Kokkos::subview(swe.depth_passive, std::make_pair(0, swe.mesh.n_vertices_host()))),
  area_active(Kokkos::subview(swe.mesh.faces.area, std::make_pair(0, swe.mesh.n_faces_host()))),
  mass_active(Kokkos::subview(swe.mass_active.view, std::make_pair(0, swe.mesh.n_faces_host()))),
  surf_passive(Kokkos::subview(swe.surf_passive, std::make_pair(0, swe.mesh.n_vertices_host()))),
  surf_active(Kokkos::subview(swe.surf_active, std::make_pair(0, swe.mesh.n_faces_host()))),
  depth_active(Kokkos::subview(sew.depth_active, std::make_pair(0, swe.mesh.n_faces_host()))),
  x_passive_1("rk4_xp1", swe.mesh.n_vertices_host()),
  x_passive_2("rk4_xp2", swe.mesh.n_vertices_host()),
  x_passive_3("rk4_xp3", swe.mesh.n_vertices_host()),
  x_passive_4("rk4_xp4", swe.mesh.n_vertices_host()),
  x_passive_work("rk4_xp_work", swe.mesh.n_vertices_host()),
  rel_vort_passive_1("rk4_zeta_p1", swe.mesh.n_vertices_host()),
  rel_vort_passive_2("rk4_zeta_p2", swe.mesh.n_vertices_host()),
  rel_vort_passive_3("rk4_zeta_p3", swe.mesh.n_vertices_host()),
  rel_vort_passive_4("rk4_zeta_p4", swe.mesh.n_vertices_host()),
  rel_vort_passive_work("rk4_zeta_pwork", swe.mesh.n_vertices_host()),
  div_passive_1("rk4_div_p1", swe.mesh.n_vertices_host()),
  div_passive_2("rk4_div_p2", swe.mesh.n_vertices_host()),
  div_passive_3("rk4_div_p3", swe.mesh.n_vertices_host()),
  div_passive_4("rk4_div_p4", swe.mesh.n_vertices_host()),
  div_passive_work("rk4_div_pwork", swe.mesh.n_vertices_host()),
  depth_passive_1("rk4_depth_p1", swe.mesh.n_vertices_host()),
  depth_passive_2("rk4_depth_p2", swe.mesh.n_vertices_host()),
  depth_passive_3("rk4_depth_p3", swe.mesh.n_vertices_host()),
  depth_passive_4("rk4_depth_p4", swe.mesh.n_vertices_host()),
  depth_passive_work("rk4_depth_pwork", swe.mesh.n_vertices_host()),
  double_dot_passive("rk4_ddot_p", swe.mesh.n_vertices_host()),
  surf_laplacian_passive("rk4_surf_lap_p", swe.mesh.n_vertices_host()),
  x_active_1("rk4_xa1", swe.mesh.n_vertices_host()),
  x_active_2("rk4_xa2", swe.mesh.n_vertices_host()),
  x_active_3("rk4_xa3", swe.mesh.n_vertices_host()),
  x_active_4("rk4_xa4", swe.mesh.n_vertices_host()),
  x_active_work("rk4_xa_work", swe.mesh.n_vertices_host()),
  rel_vort_active_1("rk4_zeta_a1", swe.mesh.n_vertices_host()),
  rel_vort_active_2("rk4_zeta_a2", swe.mesh.n_vertices_host()),
  rel_vort_active_3("rk4_zeta_a3", swe.mesh.n_vertices_host()),
  rel_vort_active_4("rk4_zeta_a4", swe.mesh.n_vertices_host()),
  rel_vort_active_work("rk4_zeta_awork", swe.mesh.n_vertices_host()),
  div_active_1("rk4_div_a1", swe.mesh.n_vertices_host()),
  div_active_2("rk4_div_a2", swe.mesh.n_vertices_host()),
  div_active_3("rk4_div_a3", swe.mesh.n_vertices_host()),
  div_active_4("rk4_div_a4", swe.mesh.n_vertices_host()),
  div_active_work("rk4_div_awork", swe.mesh.n_vertices_host()),
  area_active_1("rk4_area_a1", swe.mesh.n_vertices_host()),
  area_active_2("rk4_area_a2", swe.mesh.n_vertices_host()),
  area_active_3("rk4_area_a3", swe.mesh.n_vertices_host()),
  area_active_4("rk4_area_a4", swe.mesh.n_vertices_host()),
  area_active_work("rk4_area_awork", swe.mesh.n_vertices_host()),
  double_dot_active("rk4_ddot_a", swe.mesh.n_vertices_host()),
  surf_laplacian_active("rk4_surf_lap_a", swe.mesh.n_vertices_host()) {}

template <typename SeedType> template <typename SurfaceLaplacianType, typename BottomType>
void SWERK4<SeedType>::advance_timestep(SurfaceLaplacianType& lap_passive,
  SurfaceLaplacianType& lap_active, const BottomType& topo) {
  Kokkos::TeamPolicy<> passive_policy(swe.mesh.n_vertices_host(), Kokkos::AUTO());
  Kokkos::TeamPolicy<> active_policy(swe.mesh.n_faces_host(), Kokkos::AUTO());

  // rk stage 1: on input, velocity, double dot product, and surface laplacian
  // are already defined
  KokkosBlas::scal(x_passive_1, dt, velocity_passive);
  KokkosBlas::scal(x_active_1, dt, velocity_active);
  Kokkos::parallel_for("rk stage 1 passive", swe.mesh.n_vertices_host(),
    SWEPassiveTendencies<typename SeedType::geo>(rel_vort_passive_1,
      div_passive_1, depth_passive_1,
      x_passive, velocity_passive, rel_vort_passive, divergence_passive,
      depth_passive, double_dot_passive, surf_laplacian_passive, swe.coriolis,
      swe.g, dt));
  Kokkos::parallel_for("rk stage 1 active", swe.mesh.n_faces_host(),
    SWEActiveTendencies<typename SeedType::geo>(rel_vort_active_1,
      div_active_1, area_active_1,
      x_active, velocity_active, rel_vort_active, divergence_active,
      area_active, double_dot_active, surf_laplacian_active, coriolis,
      swe.g, dt));
  Kokkos::fence();

  // rk stage 2: set input
  KokkosBlas::update(1, x_passive, 0.5, x_passive_1, 0,  x_passive_work);
  KokkosBlas::update(1, rel_vort_passive, 0.5, rel_vort_passive_1, 0, rel_vort_passive_work);
  KokkosBlas::update(1, divergence_passive, 0.5, div_passive_1, 0, div_passive_work);
  KokkosBlas::update(1, depth_passive, 0.5, depth_passive_1, 0, depth_passive_work);

  KokkosBlas::update(1, x_active, 0.5, x_active_1, 0,  x_active_work);
  KokkosBlas::update(1, rel_vort_active, 0.5, rel_vort_active_1, 0, rel_vort_active_work);
  KokkosBlas::update(1, divergence_active, 0.5, div_active_1, 0, div_active_work);
  KokkosBlas::update(1, area_active, 0.5, area_active_1, 0, area_active_work);

  // update fluid surface & depth
  Kokkos::parallel_for("rk stage 2 surface update (passive)", swe.mesh.n_vertices_host(),
    SurfaceUpdatePassive(surf_passive, depth_passive_work, x_passive_work, topo));
  Kokkos::parallel_for("rk stage 2 surface update (active)", swe.mesh.n_faces_host(),
    SurfaceUpdateActive(surf_active, depth_active, mass_active, area_active_work,
      swe.mesh.faces.mask, x_active_work, topo));
  // compute surface laplacian
  lap_passive.compute();
  lap_active.compute();

  // compute velocity

}

} // namespace Lpm

#endif
