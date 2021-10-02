#include <bitset>
#include <iostream>
#include "LpmConfig.h"
#include "lpm_coords.hpp"
#include "lpm_logger.hpp"
#include "lpm_comm.hpp"
#include "lpm_constants.hpp"
#include "tree/lpm_tree_defs.hpp"
#include "tree/lpm_tree_common.hpp"
#include "tree/lpm_gpu_octree_functions.hpp"
#include "util/lpm_math.hpp"
#include "util/lpm_floating_point.hpp"
#include "catch.hpp"

using namespace Lpm;
using namespace Lpm::octree;

TEST_CASE("gpu_octree_functions", "[tree]") {

  Comm comm;

  Logger<> logger("gpu_octree0", Log::level::info, comm);

  const auto nkeys = 64;
  std::vector<key_type> leaf_keys(nkeys);
  Kokkos::View<Real[64][3],Host> leaf_centroids("leaf_centroids");

  REQUIRE( std_box() == box_from_key(0,0));

  for (key_type k=0; k<nkeys; ++k) {
    Real cx = 0;
    Real cy = 0;
    Real cz = 0;
    Real half_len = 1;
    for (auto l=1; l<=2; ++l) {
      half_len *= 0.5;
      const auto lkey = local_key(k, l, 2);
      cz += ( (lkey&1) ? half_len : -half_len);
      cy += ( (lkey&2) ? half_len : -half_len);
      cx += ( (lkey&4) ? half_len : -half_len);
    }
    leaf_centroids(k,0) = cx;
    leaf_centroids(k,1) = cy;
    leaf_centroids(k,2) = cz;
  }


  SECTION("key/box tests") {
    const int max_depth = 2;

    using bits = std::bitset<6>;

    for (auto i=0; i<nkeys; ++i) {
      const auto cxyz = Kokkos::subview(leaf_centroids, i, Kokkos::ALL);
      const auto key = compute_key_for_point(cxyz, max_depth);
      const auto code = encode(key, i);
      const auto key_decode = decode_key(code);
      const auto id_decode = decode_id(code);
      REQUIRE(key == key_decode);
      REQUIRE(id_decode == i);
      REQUIRE(key == i);
      const auto box = box_from_key(key, max_depth, max_depth);
      logger.debug("key = {} pt = ({}, {}, {}) box = {}", bits(key),
         cxyz(0), cxyz(1), cxyz(2), box);
      REQUIRE(box.contains_pt(cxyz));
      const auto cntd = box.centroid();
      REQUIRE(FloatingPoint<Real>::equiv(cntd[0], cxyz(0)));
      REQUIRE(FloatingPoint<Real>::equiv(cntd[1], cxyz(1)));
      REQUIRE(FloatingPoint<Real>::equiv(cntd[2], cxyz(2)));
    }

    logger.info("keys/box tests pass.");
  }

  SECTION("key/node/parent tests") {
    const int max_depth = 4;

    using bits = std::bitset<12>;

    for (auto i=0; i<nkeys; ++i) {
      const auto cxyz = Kokkos::subview(leaf_centroids, i, Kokkos::ALL);
      const auto key = compute_key_for_point(cxyz, max_depth);
      const auto parent = parent_key(key, max_depth, max_depth);
      const auto grandparent = parent_key(key, max_depth-1, max_depth);
      const auto loc_key = local_key(key, max_depth, max_depth);
      const auto parent_local = local_key(parent, max_depth-1, max_depth);
      const auto build_key = node_key(parent, loc_key, max_depth, max_depth);
      const auto build_parent = node_key(grandparent, parent_local, max_depth-1, max_depth);
      logger.debug("k = {} parent = {} grandparent = {}",
        bits(key), bits(parent), bits(grandparent));
      REQUIRE(key == build_key);
      REQUIRE(parent == build_parent);
    }

    logger.info("key/node/parent tests pass.");

  }

  SECTION("binary search tests") {

    Kokkos::View<unsigned[8],Host> arr_view("arr_view");
    for (auto i=0; i<8; ++i) {
      arr_view(i) = i;
    }
    arr_view(4) = 3;
    arr_view(5) = 3;

    logger.debug("starting binary test loop.");

    for (auto i=0; i<8; ++i) {
      const auto first = binary_search_first(i, arr_view);
      const auto last = binary_search_last(i, arr_view);
      logger.debug("found first {} at idx {}", i, first);
      logger.debug("found last  {} at idx {}", i, last);
      REQUIRE(last >= first);
      if (i == 3) {
        REQUIRE(first == 3);
        REQUIRE(last == 5);
      }
      else if (i == 4 or i==5) {
        REQUIRE(first == constants::NULL_IND);
        REQUIRE(last == constants::NULL_IND);
      }
      else {
        REQUIRE(first >= 0);
      }
    }

    logger.info("binary search tests passs.");
  }

}
