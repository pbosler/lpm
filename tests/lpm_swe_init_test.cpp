#include "LpmConfig.h"
#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
#include "lpm_swe.hpp"
#include "lpm_swe_impl.hpp"
#include "lpm_swe_gallery.hpp"
#include "mesh/lpm_mesh_seed.hpp"
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

using namespace Lpm;

TEMPLATE_TEST_CASE("planar_swe", "[swe]", QuadRectSeed, TriHexSeed) {
  // typedef PlanarGravityWaveFreeBoundaries ic_type;
  using ic_type = NitscheStricklandVortex;
  using seed_type = TestType;
  using Coriolis = CoriolisBetaPlane;

  int tree_init_depth = 4;
  Real radius         = 6;
  int amr_limit       = 0;

  Real init_plane_f0   = 0;
  Real init_plane_beta = 0;

  Comm comm;
  std::string test_name = "planar_swe_" + seed_type::id_string();
  Logger<> logger(test_name, Log::level::debug, comm);

  Coriolis coriolis(init_plane_f0, init_plane_beta);

  PolyMeshParameters<seed_type> mesh_params(tree_init_depth, radius, amr_limit);
  auto plane = std::make_unique<SWE<seed_type>>(mesh_params, coriolis);

  ic_type ic;
  plane->init_swe_problem(ic);

  auto vtk = vtk_mesh_interface(*plane);
  vtk.write(test_name + vtp_suffix());

}

TEMPLATE_TEST_CASE("sphere swe", "[swe]", CubedSphereSeed, IcosTriSphereSeed) {
  using ic_type = SWETestCase2;
  using seed_type = TestType;
  using Coriolis = CoriolisSphere;

  int tree_init_depth = 5;
  Comm comm;
  std::string test_name = "sphere_swe_" + seed_type::id_string();
  Logger<> logger(test_name, Log::level::debug, comm);

  ic_type ic;

  PolyMeshParameters<seed_type> mesh_params(tree_init_depth);
  Coriolis coriolis(ic.Omega);
  auto sphere = std::make_unique<SWE<seed_type>>(mesh_params, coriolis);
  sphere->init_swe_problem(ic);


  auto vtk = vtk_mesh_interface(*sphere);
  vtk.write(test_name + vtp_suffix());

}
