#include "LpmConfig.h"
#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
#include "lpm_geometry.hpp"
#include "lpm_coords.hpp"
#include "mesh/lpm_vertices.hpp"
#include "catch.hpp"
#include <vector>

using namespace Lpm;

TEST_CASE ("vertices", "[mesh]") {

  Comm comm;

  Logger<> logger("vertices_test", Log::level::info, comm);

  const Int nmax = 10;
  const Int nverts = 5;

  const std::vector<Index> crd_idxs = {2, 3, 4, 5, 6};

  SECTION("basic vertices") {
    logger.trace("basic_vertices");

    Vertices<Coords<PlaneGeometry>> verts(nmax);
    logger.trace("returned from constructor");
    logger.info(verts.info_string());

    REQUIRE(!verts.verts_are_dual());

    REQUIRE(verts.n_max() == nmax);
    REQUIRE(verts.nh() == 0);

    for (int v=0; v<nverts; ++v) {
      verts.insert_host(crd_idxs[v]);
    }
    REQUIRE(verts.nh() == nverts);
  }

  SECTION("dual vertices") {

    const bool verts_are_dual = true;
    Vertices<Coords<PlaneGeometry>> verts(nmax, verts_are_dual);

    REQUIRE(verts.verts_are_dual());

    const int vertex_degree = 3;
    for (int v=0; v<nverts; ++v) {
      std::vector<Index> edges_at_v(vertex_degree);
      for (int e=0; e<vertex_degree; ++e) {
        edges_at_v[e] = 10*v + e;
      }
      verts.insert_host(crd_idxs[v],edges_at_v);
    }

    REQUIRE(verts.nh() == nverts);

  }
}
