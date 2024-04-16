#ifndef LPM_DFS_BVE_SOLVER_IMPL_HPP
#define LPM_DFS_BVE_SOLVER_IMPL_HPP

#include "lpm_dfs_bve_solver.hpp"
#include "lpm_bve_sphere_kernels.hpp"
#include "lpm_compadre.hpp"
#include <KokkosBlas.hpp>

namespace Lpm {
namespace DFS {

template <typename SeedType>
void DFSRK2<SeedType>::interpolate_vorticity_from_mesh_to_grid(scalar_view_type& rel_vort_grid, const crd_view& xyz_mesh, const crd_view& xyz_grid, const scalar_view_type& rel_vort_mesh) const {
  const auto gmls_ops = std::vector<Compadre::TargetOperation>({Compadre::ScalarPointEvaluation});
  auto rel_vort_gmls = gmls::sphere_scalar_gmls(xyz_mesh, xyz_grid, sphere.mesh_to_grid_neighborhoods, sphere.gmls_params, gmls_ops);

  Compadre::Evaluator rel_vort_eval(&rel_vort_gmls);
  rel_vort_grid = rel_vort_eval.applyAlphasToDataAllComponentsAllTargetSites<Real*,DevMemory>(
    rel_vort_mesh, Compadre::ScalarPointEvaluation, Compadre::PointSample);
}

template <typename SeedType>
void DFSRK2<SeedType>::advance_timestep()  {

  // rk stage 1: vorticity
  Kokkos::parallel_for("rk2 stage1: vorticity", rel_vort_particles.extent(0),
    BVEVorticityTendency(rel_vort_particles1, velocity_particles, dt, Omega));
  // rk stage 1: positions
  KokkosBlas::axpby(dt, velocity_particles, 0, xyz_particles1);

  // input for stage 2
  KokkosBlas::update(1, rel_vort_particles, 1, rel_vort_particles1, 0, rel_vort_particles_work);
  KokkosBlas::update(1, xyz_particles, 1, xyz_particles1, 0, xyz_particles_work);

  // rk stage 2: update vorticity on dfs grid
  interpolate_vorticity_from_mesh_to_grid(rel_vort_grid, xyz_particles_work,
    xyz_grid, rel_vort_particles_work);
  // rk stage 2: solve for velocity at particles
  dfs_vort_2_velocity(xyz_particles_work, rel_vort_grid, vel_particles);
  // rk stage 2: vorticity
  Kokkos::parallel_for("rk stage2: vorticity", rel_vort_particles.extent(0),
    BVEVorticityTendency(rel_vort_particles2, vel_particles, dt, Omega));
  // rk stage 2: positions
  KokkosBlas::axpby(dt, vel_particles, 0, xyz_particles2);

  // rk update: set new particle positions
  KokkosBlas::update(0.5, xyz_particles1, 0.5, xyz_particles2, 1, xyz_particles);
  // rk update: set new vorticity
  KokkosBlas::update(0.5, rel_vort_particles1, 0.5, rel_vort_particles2, 1, rel_vort_particles);

  // update velocity to the particles
  interpolate_vorticity_from_mesh_to_grid(rel_vort_grid, xyz_particles,
    xyz_grid, rel_vort_particles);

  dfs_vort_2_velocity(xyz_particles, rel_vort_grid, velocity_particles);

  sphere.rel_vort_grid.view = rel_vort_grid;

  ++t_idx;
}

} // namespace DFS
} // namespace Lpm

#endif
