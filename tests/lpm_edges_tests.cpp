#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <sstream>

#include "LpmConfig.h"
#include "lpm_comm.hpp"
#include "lpm_coords.hpp"
#include "lpm_geometry.hpp"
#include "lpm_logger.hpp"
#include "mesh/lpm_edges.hpp"
#include "mesh/lpm_vertices.hpp"
#include "mesh/lpm_vertices_impl.hpp"
#include "util/lpm_floating_point.hpp"

using namespace Lpm;

typedef ko::DefaultExecutionSpace ExeSpace;

TEST_CASE("edges test", "[mesh]") {
  Comm comm;
  Logger<> logger("edges_test", Log::level::info, comm);

  SECTION("basic functions") {
    const int nmax_verts = 6;
    const int nmax_edges = 6;

    using crd_type  = Coords<SphereGeometry>;
    using vert_type = Vertices<crd_type>;

    // create the vertices
    auto verts = vert_type(nmax_verts);

    // create 4 coordinates for a simple Vertices object
    const Real p0[3] = {0.57735026918962584, -0.57735026918962584,
                        0.57735026918962584};
    const Real p1[3] = {0.57735026918962584, -0.57735026918962584,
                        -0.57735026918962584};
    const Real p2[3] = {0.57735026918962584, 0.57735026918962584,
                        -0.57735026918962584};
    const Real p3[3] = {0.57735026918962584, 0.57735026918962584,
                        0.57735026918962584};
    verts.insert_host(p0, p0);
    verts.insert_host(p1, p1);
    verts.insert_host(p2, p2);
    verts.insert_host(p3, p3);
    verts.update_device();

    REQUIRE(verts.nh() == 4);

    // create a set of edges with the above vertices
    Edges edges(6);
    const Index e0[4] = {0, 1, 0, 3};
    const Index e1[4] = {1, 2, 0, 5};
    const Index e2[4] = {2, 3, 0, 1};
    REQUIRE(edges.n_max() == 6);
    REQUIRE(edges.nh() == 0);
    logger.info(edges.info_string("init"));

    edges.insert_host(e0[0], e0[1], e0[2], e0[3]);
    edges.insert_host(e1[0], e1[1], e1[2], e1[3]);
    edges.insert_host(e2[0], e2[1], e2[2], e2[3]);
    REQUIRE(edges.nh() == 3);

    logger.info(edges.info_string("insert"));

    logger.debug("calling divide");
    edges.divide(0, verts);
    logger.debug("returned from divide");
    edges.update_device();
    verts.update_device();
    REQUIRE(edges.nh() == 5);
    REQUIRE(edges.n_leaves_host() == 4);
    logger.info("sph. crds after divide:\n{}", verts.info_string("after"));
    logger.info("edges after divide:\n{}", edges.info_string("after divide"));

    REQUIRE(edges.has_kids_host(0));
    REQUIRE(edges.kid_host(0, 0) == 3);
    REQUIRE(edges.kid_host(0, 1) == 4);
    REQUIRE(edges.parent_host(3) == 0);
    REQUIRE(edges.parent_host(4) == 0);
    REQUIRE(((edges.left_host(0) == edges.left_host(3)) and
             (edges.right_host(0) == edges.right_host(3))));
    REQUIRE(((edges.left_host(0) == edges.left_host(4)) and
             (edges.right_host(0) == edges.right_host(4))));

    REQUIRE(!edges.has_kids_host(1));

    REQUIRE(!edges.has_kids_host(4));
  }

  SECTION("init from seed") {
    Edges sedges(14);
    const MeshSeed<QuadRectSeed> seed;
    sedges.init_from_seed(seed);
    logger.info("seed_init:\n{}", sedges.info_string("QuadRectSeed"));
  }
}
