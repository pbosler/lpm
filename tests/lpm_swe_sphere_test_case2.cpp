#include "LpmConfig.h"
#include "lpm_comm.hpp"
#include "lpm_constants.hpp"
#include "lpm_coriolis.hpp"
#include "lpm_geometry.hpp"
#include "lpm_input.hpp"
#include "lpm_logger.hpp"
#include "lpm_pse.hpp"
#include "lpm_surface_gallery.hpp"
#include "lpm_swe.hpp"
#include "lpm_swe_gallery.hpp"
#include "lpm_swe_impl.hpp"
#include "mesh/lpm_mesh_seed.hpp"
#include "util/lpm_string_util.hpp"
#include "util/lpm_timer.hpp"
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

using namespace Lpm;

TEMPLATE_TEST_CASE ("sphere_test_case2_init", "[swe]", CubedSphereSeed, IcosTriSphereSeed) {
  using Geometry = SphereGeometry;
  using Coriolis = CoriolisSphere;
  using SeedType = TestType;

  Comm comm;
  const std::string test_name = "sphere_tc2";
  Logger<> logger(test_name, Log::level::debug, comm);

  int tree_depth = 2;
  constexpr Real sphere_radius = 1.0;
  constexpr Real h0 = 5.0;
  constexpr Real g = 1.0;
  constexpr Real u0 = constants::PI/6;
  constexpr Real Omega = 2*constants::PI;
  Coriolis coriolis(Omega);

  SWETestCase2 test_case2(h0, Omega, u0, g);
  const Real Ro = rossby_number(u0, 2*Omega, sphere_radius);
  const Real Fr = froude_number(u0, g, h0);
  logger.info(test_case2.info_string());
  logger.info("SWETestCase2 initialized with Rossby number = {}, Froude number = {}", Ro, Fr);

  PolyMeshParameters<SeedType> mesh_params(tree_depth);

  auto sphere = std::make_unique<SWE<SeedType>>(mesh_params, coriolis);

  sphere->init_swe_problem(test_case2);

  logger.debug(sphere->mass_active.info_string());

  logger.info(sphere->info_string());

  auto vtk = vtk_mesh_interface(*sphere);
  vtk.write(test_name + SeedType::id_string() + vtp_suffix());
}
