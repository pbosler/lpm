#include "dfs/lpm_dfs_bve.hpp"
#include "dfs/lpm_dfs_bve_impl.hpp"
#include "dfs/lpm_dfs_bve_solver.hpp"
#include "dfs/lpm_dfs_bve_solver_impl.hpp"
#include "dfs/lpm_dfs_grid.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
#include "lpm_compadre.hpp"
#include "lpm_vorticity_gallery.hpp"
#include "lpm_error.hpp"
#include "lpm_error_impl.hpp"
#include <catch2/catch_test_macros.hpp>

using namespace Lpm;

TEST_CASE("dfs_bve_unit_tests", "[dfs]") {
  Comm comm;
  Logger<> logger("dfs_bve_tests", Log::level::debug, comm);

  const Int nlon = 10;
  const Int ntracers = 0;

  typedef CubedSphereSeed seed_type;
  RossbyWave54Velocity velocity_fn;
  RossbyHaurwitz54 vorticity_fn;

  const Int mesh_depth = 3;
  PolyMeshParameters<seed_type> mesh_params(mesh_depth);
  const Int gmls_order = 4;
  gmls::Params gmls_params(gmls_order);

  DFS::DFSBVE<seed_type> dfs_bve(mesh_params, nlon, gmls_params);
  dfs_bve.init_vorticity(vorticity_fn);
  dfs_bve.init_velocity(velocity_fn);

  // mesh to grid
  ScalarField<VertexField> rel_vort_grid_gmls("relative_vorticity_gmls",
    dfs_bve.vtk_grid_size());
  dfs_bve.interpolate_vorticity_from_mesh_to_grid(rel_vort_grid_gmls);

  logger.info(dfs_bve.info_string());

  scalar_view_type rel_vort_error("relative_vorticity_gmls_error", dfs_bve.grid_size());
  ErrNorms gmls_err(rel_vort_error, rel_vort_grid_gmls.view, dfs_bve.rel_vort_grid.view, dfs_bve.grid_area);
  logger.info(gmls_err.info_string());

  // grid to mesh
	dfs_bve.interpolate_velocity_from_grid_to_mesh();

  // finish-up: write output
  const std::string mesh_vtk_file = "dfs_bve_particles.vtp";
  const std::string grid_vtk_file = "dfs_bve_grid.vts";

  auto vtk_mesh = vtk_mesh_interface(dfs_bve);
  auto vtk_grid = vtk_grid_interface(dfs_bve);
  vtk_grid.add_scalar_point_data(rel_vort_grid_gmls.view, "relative_vorticity_gmls");
  vtk_grid.add_scalar_point_data(rel_vort_error, "relative_vorticity_gmls_error");

  vtk_mesh.write(mesh_vtk_file);
  vtk_grid.write(grid_vtk_file);

}


