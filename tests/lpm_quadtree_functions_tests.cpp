#include <bitset>
#include <iostream>
#include "LpmConfig.h"
#include "lpm_coords.hpp"
#include "lpm_logger.hpp"
#include "lpm_comm.hpp"
#include "lpm_constants.hpp"
#include "tree/lpm_box2d.hpp"
#include "tree/lpm_quadtree_functions.hpp"
#include "util/lpm_math.hpp"
#include "util/lpm_floating_point.hpp"
#include "catch.hpp"

using namespace Lpm;
using namespace quadtree;

TEST_CASE("quadtree_functions", "[tree]") {

  Comm comm;

  Logger<> logger("gpu_octree0", Log::level::info, comm);

  const auto nkeys = 16;
  std::vector<key_type> leaf_keys(nkeys);
  Kokkos::View<Real[16][2],Host> leaf_centroids("leaf_centroids");

  REQUIRE(default_box() == box_from_key(0,0));
}
