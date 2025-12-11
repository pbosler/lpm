#ifndef LPM_DFS_POLAR_VORTEX_SOLVER_IMPL_HPP
#define LPM_DFS_POLAR_VORTEX_SOLVER_IMPL_HPP

#include "lpm_dfs_polar_vortex_solver.hpp"
#include "lpm_vorticity_gallery.hpp"
#include "lpm_bve_sphere_kernels.hpp"
#include "lpm_geometry.hpp"

#include <KokkosBlas.hpp>

namespace Lpm {
namespace DFS {

namespace impl {
struct RK4Update {
  using crd_view = typename SphereGeometry::crd_view_type;

  static constexpr Real sixth = 1.0 / 6.0;
  static constexpr Real third = 1.0 / 3.0;

  crd_view x, x1, x2, x3, x4;

  scalar_view_type vort, vort1, vort2, vort3, vort4;

  RK4Update(crd_view& x_inout, const crd_view& x1, const crd_view& x2,
    const crd_view& x3, const crd_view& x4,
    scalar_view_type vort_inout, const scalar_view_type& z1, const scalar_view_type& z2,
    const scalar_view_type& z3, const scalar_view_type& z4) :
    x(x_inout), x1(x1), x2(x2), x3(x3), x4(x4),
    vort(vort_inout), vort1(z1), vort2(z2), vort3(z3), vort4(z4) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index& i) const {
    for (int j=0; j<3; ++j) {
      x(i,j) += sixth *(x1(i,j) + x4(i,j)) + third * (x2(i,j) + x3(i,j));
    }
    vort(i) += sixth*(vort1(i)+vort4(i)) + third*(vort2(i) + vort3(i));
  }
};
} // namespace impl

template <typename SeedType>
void DFSPolarVortexRK4<SeedType>::advance_timestep() {

  // rk stage 1 vorticity
  //  rel_vort_particles1 = dt * d\zeta / dt
  Kokkos::parallel_for("rk4 stage 1 vorticity", rel_vort_particles.extent(0),
    BVEPolarVortexVorticityTendency(rel_vort_particles1, xyz_particles, velocity_particles,
      t, dt, Omega));
  // rk stage 1 positions
  //    xyz_particles1 = dt * velocity_particles
  KokkosBlas::axpby(dt, velocity_particles, 0, xyz_particles1);

  // input for stage 2
  //    rel_vort_particles_work = rel_vort_particles + 0.5 * rel_vort_particles1
  KokkosBlas::update(1.0, rel_vort_particles, 0.5, rel_vort_particles, 0, rel_vort_particles_work);
  //    xyz_particles_work = xyz_particles + 0.5 * xyz_particles1
  KokkosBlas::update(1.0, xyz_particles, 0.5, xyz_particles1, 0, xyz_particles_work);
  normalize_coordinates(xyz_particles_work);

  // rk stage 2: update vorticity on dfs grid
  interpolate_vorticity_from_mesh_to_grid(rel_vort_grid, xyz_particles_work,
    xyz_grid, rel_vort_particles_work);
  // rk stage 2: solve for velocity at particles
  dfs_vort_2_velocity(xyz_particles_work, rel_vort_grid, vel_particles);
  // rk stage 2: vorticity
  Kokkos::parallel_for("rk4 stage2 vorticity", rel_vort_particles.extent(0),
    BVEPolarVortexVorticityTendency(rel_vort_particles2, xyz_particles_work, vel_particles, t+0.5*dt, dt, Omega));
  // rk stage 2: positions
  KokkosBlas::axpby(dt, vel_particles, 0.0, xyz_particles2);

  // input for stage 3
  KokkosBlas::update(1.0, rel_vort_particles, 0.5, rel_vort_particles2, 0.0, rel_vort_particles_work);
  KokkosBlas::update(1.0, xyz_particles, 0.5, xyz_particles2, 0.0, xyz_particles_work);
  normalize_coordinates(xyz_particles_work);

  // rk stage 3: update vorticity on dfs grid
  interpolate_vorticity_from_mesh_to_grid(rel_vort_grid, xyz_particles_work,
    xyz_grid, rel_vort_particles_work);
  // rk stage 3: solve for velocity update at particles
  dfs_vort_2_velocity(xyz_particles_work, rel_vort_grid, vel_particles);
  // rk stage 3: vorticity
  Kokkos::parallel_for("rk4 stage 3 vorticity", rel_vort_particles.extent(0),
    BVEPolarVortexVorticityTendency(rel_vort_particles3, xyz_particles_work, vel_particles, t+0.5*dt, dt, Omega));
  // rk stage 3: positions
  KokkosBlas::axpby(dt, vel_particles, 0.0, xyz_particles3);

  // input for stage 4
  KokkosBlas::update(1.0, rel_vort_particles, 1.0, rel_vort_particles3, 0.0, rel_vort_particles_work);
  KokkosBlas::update(1.0, xyz_particles, 1.0, xyz_particles3, 0.0, xyz_particles_work);
  normalize_coordinates(xyz_particles_work);

  // rk stage 4: update vorticity on dfs grid
  interpolate_vorticity_from_mesh_to_grid(rel_vort_grid, xyz_particles_work,
    xyz_grid, rel_vort_particles_work);
  // rk stage 4: solve for velocity update at particles
  dfs_vort_2_velocity(xyz_particles_work, rel_vort_grid, vel_particles);
  // rk stage 4: vorticity
  Kokkos::parallel_for("rk4 stage 4 vorticity", rel_vort_particles.extent(0),
    BVEPolarVortexVorticityTendency(rel_vort_particles4, xyz_particles_work, vel_particles, t + dt, dt, Omega));
  // rk stage 4: positions
  KokkosBlas::axpby(dt, vel_particles, 0.0, xyz_particles4);


  Kokkos::parallel_for("rk4 update", rel_vort_particles.extent(0),
    impl::RK4Update(xyz_particles, xyz_particles1, xyz_particles2, xyz_particles3, xyz_particles4, rel_vort_particles, rel_vort_particles1, rel_vort_particles2, rel_vort_particles3, rel_vort_particles4));

  Kokkos::fence();

  // set up for next time step
  normalize_coordinates(xyz_particles);
  interpolate_vorticity_from_mesh_to_grid(rel_vort_grid, xyz_particles, xyz_grid, rel_vort_particles);
  dfs_vort_2_velocity(xyz_particles, rel_vort_grid, velocity_particles);
  sphere.rel_vort_grid.view = rel_vort_grid;

  t = (t_idx++) * dt;
}

template <typename SeedType>
void DFSPolarVortexRK4<SeedType>::normalize_coordinates(crd_view xyz) {
  Kokkos::parallel_for("normalize", xyz.extent(0), SphereNormalize(xyz));
}

template <typename SeedType>
void DFSPolarVortexRK4<SeedType>::interpolate_vorticity_from_mesh_to_grid(scalar_view_type& rel_vort_grid, const crd_view& xyz_mesh, const crd_view& xyz_grid, const scalar_view_type& rel_vort_mesh) const {
  const auto gmls_ops = std::vector<Compadre::TargetOperation>({Compadre::ScalarPointEvaluation});
  auto rel_vort_gmls = gmls::sphere_scalar_gmls(xyz_mesh, xyz_grid, sphere.mesh_to_grid_neighborhoods, sphere.gmls_params, gmls_ops);

  Compadre::Evaluator rel_vort_eval(&rel_vort_gmls);
  rel_vort_grid = rel_vort_eval.applyAlphasToDataAllComponentsAllTargetSites<Real*,DevMemory>(
    rel_vort_mesh, Compadre::ScalarPointEvaluation, Compadre::PointSample);
}

} // namespace DFS
} // namespace Lpm
#endif
