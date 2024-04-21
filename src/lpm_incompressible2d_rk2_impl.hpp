#ifndef LPM_INCOMPRESSIBLE2D_RK2_IMPL_HPP
#define LPM_INCOMPRESSIBLE2D_RK2_IMPL_HPP

#include "LpmConfig.h"
#include "lpm_incompressible2d.hpp"
#include "lpm_incompressible2d_impl.hpp"
#include "lpm_incompressible2d_kernels.hpp"
#include "lpm_incompressible2d_rk2.hpp"
#include "util/lpm_string_util.hpp"

#include <KokkosBlas.hpp>

namespace Lpm {

template <typename SeedType>
Incompressible2DRK2<SeedType>::Incompressible2DRK2(const Real dt,
  Incompressible2D<SeedType>& ic2d) :
  dt(dt),
  ic2d(ic2d),
  t_idx(0),
  n_passive(ic2d.mesh.n_vertices_host()),
  n_active(ic2d.mesh.n_faces_host()),
  eps(ic2d.eps),
  passive_x1("passive_x1", ic2d.mesh.n_vertices_host()),
  passive_x2("passive_x1", ic2d.mesh.n_vertices_host()),
  passive_xwork("passive_x1", ic2d.mesh.n_vertices_host()),
  passive_rel_vort1("passive_rel_vort1", ic2d.mesh.n_vertices_host()),
  passive_rel_vort2("passive_rel_vort1", ic2d.mesh.n_vertices_host()),
  passive_rel_vortwork("passive_rel_vort1", ic2d.mesh.n_vertices_host()),
  active_x1("active_x1", ic2d.mesh.n_faces_host()),
  active_x2("active_x1", ic2d.mesh.n_faces_host()),
  active_xwork("active_x1", ic2d.mesh.n_faces_host()),
  active_rel_vort1("active_rel_vort1", ic2d.mesh.n_faces_host()),
  active_rel_vort2("active_rel_vort1", ic2d.mesh.n_faces_host()),
  active_rel_vortwork("active_rel_vort1", ic2d.mesh.n_faces_host())
{
  passive_policy = std::make_unique<Kokkos::TeamPolicy<>>(ic2d.mesh.n_vertices_host(), Kokkos::AUTO());
  active_policy = std::make_unique<Kokkos::TeamPolicy<>>(ic2d.mesh.n_faces_host(), Kokkos::AUTO());
}

template <typename SeedType>
void Incompressible2DRK2<SeedType>::advance_timestep_impl() {

  // rk stage 1: position
  KokkosBlas::scal(passive_x1, dt, ic2d.velocity_passive.view);
  KokkosBlas::scal(active_x1, dt, ic2d.velocity_active.view);

  // rk stage 1: vorticity
  Kokkos::parallel_for("Incompressible2DRK2-1 passive tendencies",
    n_passive,
    Incompressible2DTendencies<geo>(passive_rel_vort1,
      ic2d.velocity_passive.view,
      ic2d.coriolis));
  Kokkos::parallel_for("Incompressible2DRK2-1 active tendencies",
    n_active,
    Incompressible2DTendencies<geo>(active_rel_vort1,
      ic2d.velocity_active.view,
      ic2d.coriolis));

  // input for stage 2: vorticity
  KokkosBlas::update(1, ic2d.rel_vort_passive.view,
                    dt, passive_rel_vort1,
                     0, passive_rel_vortwork);
  KokkosBlas::update(1, ic2d.rel_vort_active.view,
                    dt, active_rel_vort1,
                     0, active_rel_vortwork);
  // input for stage 2: position
  KokkosBlas::update(1, ic2d.mesh.vertices.phys_crds.view,
                    dt, ic2d.velocity_passive.view,
                     0, passive_xwork);
  KokkosBlas::update(1, ic2d.mesh.faces.phys_crds.view,
                    dt, ic2d.velocity_active.view,
                     0, active_xwork);

  // rk stage 2: velocity
  Kokkos::parallel_for("Incompressible2DRK2-1 velocity, passive",
    *passive_policy,
    Incompressible2DPassiveSums<geo>(ic2d.velocity_passive.view,
      ic2d.stream_fn_passsive.view,
      passive_xwork,
      active_xwork,
      active_rel_vortwork,
      ic2d.mesh.faces.area,
      ic2d.mesh.faces.mask,
      eps,
      n_active));
  Kokkos::parallel_for("Incompressible2DRK2-1 velocity, active",
    *active_policy,
    Incompressible2DActiveSums<geo>(ic2d.velocity_active.view,
      ic2d.stream_fn_active.view,
      active_xwork,
      active_rel_vortwork,
      ic2d.mesh.faces.area,
      ic2d.mesh.faces.mask,
      eps,
      n_active));

  // rk stage 2: position
  KokkosBlas::scal(passive_x2, dt, ic2d.velocity_passive.view);
  KokkosBlas::scal(active_x2, dt, ic2d.velocity_active.view);

  // rk stage 2: vorticity
  Kokkos::parallel_for("Incompressible2DRK2-2 passive tendencies",
    n_passive,
    Incompressible2DTendencies<geo>(passive_rel_vort2,
      ic2d.velocity_passive.view, ic2d.coriolis));
  Kokkos::parallel_for("Incompressible2DRK2-2 active tendencies",
    n_active,
    Incompressible2DTendencies<geo>(active_rel_vort2,
      ic2d.velocity_active.view, ic2d.coriolis));

  // rk2 update to next time
  KokkosBlas::update(0.5, passive_rel_vort1, 0.5, passive_rel_vort2, 1, ic2d.rel_vort_passive.view);
  KokkosBlas::update(0.5, active_rel_vort1, 0.5, active_rel_vort2, 1, ic2d.rel_vort_active.view);
  KokkosBlas::update(0.5, passive_x1, 0.5, passive_x2, 1, ic2d.mesh.vertices.phys_crds.view);
  KokkosBlas::update(0.5, active_x1, 0.5, active_x2, 1, ic2d.mesh.faces.phys_crds.view);

  Kokkos::parallel_for(*passive_policy,
    Incompressible2DPassiveSums<geo>(ic2d.velocity_passive.view,
      ic2d.stream_fn_passsive.view,
      ic2d.mesh.vertices.phys_crds.view,
      ic2d.mesh.faces.phys_crds.view,
      ic2d.rel_vort_active.view,
      ic2d.mesh.faces.area,
      ic2d.mesh.faces.mask,
      eps,
      n_active));

  Kokkos::parallel_for(*active_policy,
    Incompressible2DActiveSums<geo>(ic2d.velocity_active.view,
      ic2d.stream_fn_active.view,
      ic2d.mesh.faces.phys_crds.view,
      ic2d.rel_vort_active.view,
      ic2d.mesh.faces.area,
      ic2d.mesh.faces.mask,
      eps,
      n_active));
  ++t_idx;
}

template <typename SeedType>
std::string Incompressible2DRK2<SeedType>::info_string(const int tab_level) const {
  std::ostringstream ss;
  ss << "Incompressible2DRK2 info:\n";
  ss << "\tdt = " << dt << "\n"
     << "\tn_passive = " << n_passive << "\n"
     << "\tn_active = " << n_active << "\n"
     << "\teps = " << eps << "\n";
  return ss.str();
}

} // namespace Lpm

#endif
