#include "LpmConfig.h"
#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
#include "lpm_swe.hpp"
#include "lpm_swe_gallery.hpp"
#include "lpm_swe_impl.hpp"
#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"
#endif

#include <catch2/catch_test_macros.hpp>

using namespace Lpm;

TEST_CASE("planar_swe", "[swe]") {

typedef PlanarGravityWaveFreeBoundaries ic_type;
// typedef NitscheStricklandVortex ic_type;
typedef QuadRectSeed seed_type;

  int tree_init_depth = 4;
  Real radius = 6;
  int amr_limit = 0;

  Real init_plane_f0 = 0;
  Real init_plane_beta = 0;

  Comm comm;
  std::string test_name = "planar_swe_" + seed_type::id_string();
  Logger<> logger(test_name, Log::level::debug, comm);

  PolyMeshParameters<seed_type> mesh_params(tree_init_depth, radius, amr_limit);
  auto plane = std::make_unique<SWE<seed_type>>(mesh_params, init_plane_f0, init_plane_beta);

  ic_type ic;
//   plane->init_swe_problem(ic);
#ifdef LPM_USE_VTK
  auto vtk = vtk_mesh_interface(*plane);
  vtk.write(test_name + vtp_suffix());
#endif
}

TEST_CASE("sphere swe", "[swe]") {
  typedef SWETestCase2 ic_type;
  typedef CubedSphereSeed seed_type;

  int tree_init_depth = 5;
  Comm comm;
  std::string test_name = "sphere_swe_" + seed_type::id_string();
  Logger<> logger(test_name, Log::level::debug, comm);

  ic_type ic;

  PolyMeshParameters<seed_type> mesh_params(tree_init_depth);
  auto sphere = std::make_unique<SWE<seed_type>>(mesh_params, ic.Omega);
//   sphere->init_swe_problem(ic);

#ifdef LPM_USE_VTK
  auto vtk = vtk_mesh_interface(*sphere);
  vtk.write(test_name + vtp_suffix());
#endif
}

