#include <iostream>
#include <sstream>
#include "LpmConfig.h"
#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
#include "lpm_geometry.hpp"
#include "lpm_coords.hpp"
#include "mesh/lpm_edges.hpp"
#include "util/lpm_floating_point_util.hpp"
#include "catch.hpp"

using namespace Lpm;

typedef ko::DefaultExecutionSpace ExeSpace;

TEST_CASE("edges test", "[mesh]")  {
  Comm comm;
  Logger<> logger("edges_test", Log::level::info, comm);

  SECTION("basic functions") {
    Coords<SphereGeometry> sc4(6);
    const Real p0[3] = {0.57735026918962584,  -0.57735026918962584,  0.57735026918962584};
    const Real p1[3] = {0.57735026918962584,  -0.57735026918962584,  -0.57735026918962584};
    const Real p2[3] = {0.57735026918962584,  0.57735026918962584,  -0.57735026918962584};
    const Real p3[3] = {0.57735026918962584,  0.57735026918962584, 0.57735026918962584};
    sc4.insert_host(p0);
    sc4.insert_host(p1);
    sc4.insert_host(p2);
    sc4.insert_host(p3);
    sc4.update_device();

    logger.info("spherical crds init: {}", sc4.info_string("sc4 init"));;

    Coords<SphereGeometry> sc4lag(6);
    sc4lag.insert_host(p0);
    sc4lag.insert_host(p1);
    sc4lag.insert_host(p2);
    sc4lag.insert_host(p3);
    sc4lag.update_device();

    Edges edges(6);
    const Index e0[4] = {0,1,0,3};
    const Index e1[4] = {1,2,0,5};
    const Index e2[4] = {2,3,0,1};
    REQUIRE(edges.n_max() == 6);
    REQUIRE(edges.nh() == 0);
    logger.info(edges.info_string("init"));

    edges.insert_host(e0[0], e0[1], e0[2], e0[3]);
    edges.insert_host(e1[0], e1[1], e1[2], e1[3]);
    edges.insert_host(e2[0], e2[1], e2[2], e2[3]);
    REQUIRE(edges.nh() == 3);

    logger.info(edges.info_string("insert"));

    logger.debug("calling divide");
    edges.divide<SphereGeometry>(0, sc4, sc4lag);
    edges.update_device();
    sc4.update_device();
    sc4lag.update_device();
    REQUIRE(edges.nh() == 5);
    logger.info("sph. crds after divide:\n{}", sc4.info_string("after"));
    logger.info("edges after divide:\n{}", edges.info_string("after divide"));

    REQUIRE(edges.has_kids_host(0));

    REQUIRE(!edges.has_kids_host(1));

    REQUIRE(!edges.has_kids_host(4));
  }

  SECTION("init from seed") {
    Edges sedges(14);
    const MeshSeed<QuadRectSeed> seed;
    sedges.init_from_seed(seed);
    logger.info("seed_init:\n{}", sedges.info_string("QuadRectSeed"));
  }
  SECTION("CircularPlaneGeometry") {
    const Real p0[2] = {0,0.5};
    const Real p1[2] = {-0.5,0};
    const Real p6[2] = {-1,0};

    Coords<CircularPlaneGeometry> udcrds(8);
    udcrds.insert_host(p0);
    udcrds.insert_host(p1);
    udcrds.insert_host(p6);
    Coords<CircularPlaneGeometry> udlagcrds(8);
    udlagcrds.insert_host(p0);
    udlagcrds.insert_host(p1);
    udlagcrds.insert_host(p6);
    Edges edges(6);
    const Index e0[4] = {0,1,0,1};
    const Index e11[4] = {2,1,1,2};
    edges.insert_host(e0[0], e0[1], e0[2], e0[3]);
    edges.insert_host(e11[0], e11[1], e11[2], e11[3]);
    edges.divide<CircularPlaneGeometry>(0, udcrds, udlagcrds);
    std::cout << edges.info_string("edges after radial divide", 0, true);
    Real rmidpt[2];
    for (int i=0; i<2; ++i) rmidpt[i] = udcrds.get_crd_component_host(3,i);
    REQUIRE(FloatingPoint<Real>::equiv(CircularPlaneGeometry::mag(rmidpt), 0.5));

    edges.divide<CircularPlaneGeometry>(1, udcrds, udlagcrds);
    std::cout << edges.info_string("edges after axial divide", 0, true);
    std::cout << udcrds.info_string("crds after 2 divides", 0, true);

  }

}
