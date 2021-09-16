#include <iostream>
#include "LpmConfig.h"
#include "lpm_coords.hpp"
#include "lpm_logger.hpp"
#include "lpm_comm.hpp"
#include "lpm_geometry.hpp"
#include "mesh/lpm_mesh_seed.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#include "tree/lpm_box3d.hpp"
#include "util/lpm_floating_point.hpp"
#include "util/lpm_tuple.hpp"
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"
#include "catch.hpp"

using namespace Lpm;
using namespace Lpm::tree;

TEST_CASE("box3d", "[tree]") {
  Comm comm;

  Logger <> logger("box3d_test", Log::level::info, comm);

  SECTION("unit tests") {
    const bool padding = false;

    Box3d box0(-1,1,-1,1,-1,1, padding);
    logger.info("box0.volume() = {}", box0.volume());
    REQUIRE(FloatingPoint<Real>::equiv(box0.volume(), 8));

    logger.info("box0.aspect_ratio() = {}", box0.aspect_ratio());
    REQUIRE(box0.is_cube());
    const Box3d box00 = box0;
    Box3d box000(box0);
    REQUIRE( box0 == box00 );
    REQUIRE( box00 == box000 );

    const Real origin[3] = {0,0,0};
    REQUIRE(box0.contains_pt(origin));

    Real c0[3];
    box0.centroid(c0[0], c0[1], c0[2]);
    const auto c00 = box0.centroid();
    logger.info("box0.centroid() = {}", c00);
    REQUIRE(FloatingPoint<Real>::zero(SphereGeometry::square_euclidean_distance(c0, c00)));
    REQUIRE(FloatingPoint<Real>::zero(SphereGeometry::square_euclidean_distance(c0, origin)));



  }



}
