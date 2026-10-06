#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <memory>

#include "LpmConfig.h"
#include "dfs/lpm_dfs_bve.hpp"
#include "dfs/lpm_dfs_bve_impl.hpp"
#include "dfs/lpm_dfs_bve_solver.hpp"
#include "dfs/lpm_dfs_bve_solver_impl.hpp"
#include "dfs/lpm_dfs_grid.hpp"
#include "dfs/lpm_dfs_polar_vortex_solver.hpp"
#include "dfs/lpm_dfs_polar_vortex_solver_impl.hpp"
#include "lpm_comm.hpp"
#include "lpm_compadre.hpp"
#include "lpm_constants.hpp"
#include "lpm_logger.hpp"
#include "lpm_vorticity_gallery.hpp"
#include "mesh/lpm_polymesh2d.hpp"

using namespace Lpm;
using namespace Lpm::DFS;

/**
  Regression test for the polar vortex (Juckes & McIntyre 1986) setup.

  The relative vorticity of the initial vortex is O(1) (range ~(-1.6, 3.1)) and
  should stay O(1) for the first few tenths of a time unit. At the commit where
  this test was added, the depth-2 run below instead grows to O(10) by t ~ 0.6
  and then crashes in the GMLS vorticity-to-grid interpolation, so this test is
  EXPECTED TO FAIL until that instability is fixed.
*/
TEST_CASE("dfs_polar_vortex_stability", "[dfs]") {
  Comm comm;
  Logger<> logger("dfs_polar_vortex_stability", Log::level::info, comm);

  using SeedType = CubedSphereSeed;
  constexpr Real sphere_radius = 1.0;
  const Int mesh_depth = 2;
  const Int nlon = 20;
  const Real Omega = 2 * constants::PI;
  const Real dt = 0.025;
  const Int nsteps = 40;
  const Real vorticity_bound = 5.0;

  PolyMeshParameters<SeedType> mesh_params(mesh_depth, sphere_radius, 0, 0);
  gmls::Params gmls_params(6);
  gmls_params.manifold_order = 2;
  gmls_params.eps_multiplier = 2.2;

  auto sphere = std::make_unique<DFSBVE<SeedType>>(mesh_params, nlon, gmls_params, Omega);

  const PolarVortexParams pv_params(4 * constants::PI, 2.0, 4.0, 15.0, 6 * constants::PI / 5);
  JM86PolarVortex vorticity;
  sphere->init_vorticity(vorticity);
  vorticity.set_gauss_const(sphere->total_vorticity());
  sphere->init_vorticity(vorticity);
  sphere->finalize_mesh_to_grid_coupling();
  sphere->init_velocity_from_vorticity();

  DFSPolarVortexRK4<SeedType> solver(dt, *sphere, 0, pv_params);

  for (Int n = 0; n < nsteps; ++n) {
    sphere->advance_timestep(solver);
    const auto range = sphere->rel_vort_passive.range(sphere->mesh.n_vertices_host());
    INFO("step " << n + 1 << ", t = " << (n + 1) * dt << ", rel. vort. range = (" << range.first
                 << ", " << range.second << ")");
    REQUIRE(std::isfinite(range.first));
    REQUIRE(std::isfinite(range.second));
    REQUIRE(std::abs(range.first) < vorticity_bound);
    REQUIRE(std::abs(range.second) < vorticity_bound);
  }
}
