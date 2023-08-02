#include "dfs/lpm_dfs_bve.hpp"
#include "dfs/lpm_dfs_bve_impl.hpp"
#include "dfs/lpm_dfs_grid.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
#include "lpm_compadre.hpp"
#include "lpm_vorticity_gallery.hpp"
#include <catch2/catch_test_macros.hpp>

using namespace Lpm;

TEST_CASE("dfs_bve_tests", "[dfs]") {
  Comm comm;
  Logger<> logger("dfs_bve_tests", Log::level::debug, comm);

  const Int nlon = 90;
  const Int ntracers = 0;

  typedef CubedSphereSeed seed_type;
  RossbyHaurwitz54 vorticity_fn;

  const Int mesh_depth = 3;
  PolyMeshParameters<seed_type> mesh_params(mesh_depth);
  const Int gmls_order = 4;
  gmls::Params gmls_params(gmls_order);

  DFS::DFSBVE dfs_bve(mesh_params, nlon, ntracers, gmls_params);
  dfs_bve.init_vorticity(vorticity_fn);
  const std::string mesh_vtk_file = "dfs_bve_particles.vtp";
  const std::string grid_vtk_file = "dfs_bve_grid.vts";
  dfs_bve.write_vtk(mesh_vtk_file, grid_vtk_file);
}
