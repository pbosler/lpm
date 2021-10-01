#include <bitset>
#include <iostream>
#include "LpmConfig.h"
#include "lpm_coords.hpp"
#include "lpm_logger.hpp"
#include "lpm_comm.hpp"
#include "lpm_constants.hpp"
#include "lpm_geometry.hpp"
#include "tree/lpm_box2d.hpp"
#include "tree/lpm_tree_defs.hpp"
#include "tree/lpm_tree_common.hpp"
#include "tree/lpm_quadtree_functions.hpp"
#include "util/lpm_math.hpp"
#include "util/lpm_floating_point.hpp"
#include "catch.hpp"

using namespace Lpm;
using namespace quadtree;

TEST_CASE("quadtree_functions", "[tree]") {

  Comm comm;

  Logger<> logger("quadtree_fns", Log::level::debug, comm);

  const auto nkeys = 16;
  int max_depth = 2;
  Kokkos::View<key_type*> leaf_keys("keys", nkeys);
  Kokkos::View<Real[16][2]> leaf_centroids("leaf_centroids");
  auto h_leaf_centroids = Kokkos::create_mirror_view(leaf_centroids);
  auto h_leaf_keys = Kokkos::create_mirror_view(leaf_keys);

  REQUIRE(default_box() == box_from_key(0,0,max_depth));

  for (key_type k=0; k<nkeys; ++k) {
    Real cx = 0;
    Real cy = 0;
    Real half_len = 1;
    for (auto l=1; l<=max_depth; ++l) {
      half_len *= 0.5;
      const auto lkey = local_key(k, l, max_depth);
      cx += ( (lkey&2) ? half_len : -half_len);
      cy += ( (lkey&1) ? half_len : -half_len);
    }
    h_leaf_centroids(k,0) = cx;
    h_leaf_centroids(k,1) = cy;
    h_leaf_keys(k) = k;
  }
  Kokkos::deep_copy(leaf_centroids, h_leaf_centroids);
  Kokkos::deep_copy(leaf_keys, h_leaf_keys);

  for (int i=0; i<nkeys; ++i) {
    const auto cxy = Kokkos::subview(leaf_centroids, i, Kokkos::ALL);
    REQUIRE(h_leaf_keys(i) == compute_key_for_point(cxy, max_depth));
  }

  SECTION("key/box tests") {

    Kokkos::View<key_type*> decoded_keys("decoded_keys", nkeys);
    Kokkos::View<id_type*> decoded_ids("decoded_ids", nkeys);
    Kokkos::View<Box2d*> boxes("boxes", nkeys);

    Kokkos::parallel_for(nkeys, KOKKOS_LAMBDA (const int i) {
      const auto cxy = Kokkos::subview(leaf_centroids, i, Kokkos::ALL);
      const auto key = compute_key_for_point(cxy, max_depth);
      const auto code = encode(key, i);
      decoded_keys(i) = decode_key(code);
      decoded_ids(i) = decode_id(code);
      boxes(i) = box_from_key(key, max_depth, max_depth);
    });

    auto h_decoded_keys = Kokkos::create_mirror_view(decoded_keys);
    auto h_decoded_ids = Kokkos::create_mirror_view(decoded_ids);
    auto h_boxes = Kokkos::create_mirror_view(boxes);
    Kokkos::deep_copy(h_decoded_keys, decoded_keys);
    Kokkos::deep_copy(h_decoded_ids, decoded_ids);
    Kokkos::deep_copy(h_boxes, boxes);

    using bits = std::bitset<4>;
    for (int i=0; i<nkeys; ++i) {
      const auto cxy = Kokkos::subview(h_leaf_centroids, i, Kokkos::ALL);
      const auto key = compute_key_for_point(cxy, max_depth);
      REQUIRE(key == h_decoded_keys(i));
      REQUIRE(key == h_leaf_keys(i));
      REQUIRE(key == i);
      REQUIRE(h_decoded_ids(i) == i);
      const auto box = h_boxes(i);
      logger.debug("key {} pt ({}, {}) box {}", bits(key), cxy(0), cxy(1), box);
      REQUIRE(box.contains_pt(cxy));
      const auto cntd = box.centroid();
      REQUIRE(FloatingPoint<Real>::zero(PlaneGeometry::distance(cxy, cntd)));
    }

    logger.info("key/box tests pass.");
  }

  SECTION("key/node tests") {

    max_depth = 4;
    using bits = std::bitset<8>;

    for (int i=0; i<nkeys; ++i) {
      const auto cxy = Kokkos::subview(h_leaf_centroids, i, Kokkos::ALL);
      h_leaf_keys(i) = compute_key_for_point(cxy, max_depth);
    }
    Kokkos::deep_copy(leaf_keys, h_leaf_keys);


    Kokkos::View<key_type*> build_keys("build_keys", nkeys);
    Kokkos::View<key_type*> build_parent_keys("build_parent_keys", nkeys);
    Kokkos::View<key_type*> grandparent_keys("grandparent_keys", nkeys);

    Kokkos::parallel_for(nkeys, KOKKOS_LAMBDA (const int i) {
      const auto cxy = Kokkos::subview(leaf_centroids, i, Kokkos::ALL);
      const auto key = compute_key_for_point(cxy, max_depth);
      const auto pkey = parent_key(key, max_depth, max_depth);
      const auto gpkey = parent_key(pkey, max_depth-1, max_depth);
      const auto loc_key = local_key(key, max_depth, max_depth);
      const auto plocal = local_key(pkey, max_depth-1, max_depth);
      build_keys(i) = build_key(pkey, loc_key, max_depth, max_depth);
      build_parent_keys(i) = build_key(gpkey, plocal, max_depth-1, max_depth);
      grandparent_keys(i) = gpkey;
    });

    auto h_build_keys = Kokkos::create_mirror_view(build_keys);
    auto h_build_parent_keys = Kokkos::create_mirror_view(build_parent_keys);
    auto h_grandparent_keys = Kokkos::create_mirror_view(grandparent_keys);
    Kokkos::deep_copy(h_build_keys, build_keys);
    Kokkos::deep_copy(h_build_parent_keys, build_parent_keys);
    Kokkos::deep_copy(h_grandparent_keys, grandparent_keys);

    for (int i=0; i<nkeys; ++i) {
      logger.debug("i = {}: k = {}, kb = {} parent = {}, grandparent = {}", i,
        bits(h_leaf_keys(i)), bits(h_build_keys(i)), bits(h_build_parent_keys(i)), bits(h_grandparent_keys(i)));
      REQUIRE(h_leaf_keys(i) == h_build_keys(i));
    }

    logger.info("key/node tests pass.");
  }

  SECTION("binary search tests") {
    Kokkos::View<unsigned[8]> arr_view("arr_view");
    auto h_arr_view = Kokkos::create_mirror_view(arr_view);
    for (int i=0; i<arr_view.extent(0); ++i) {
      h_arr_view(i) = i;
    }
    h_arr_view(4) = 3;
    h_arr_view(5) = 3;

    Kokkos::deep_copy(arr_view, h_arr_view);

    Kokkos::View<Int*> firsts("firsts", arr_view.extent(0));
    Kokkos::View<Int*> lasts("lasts", arr_view.extent(0));

    Kokkos::parallel_for(arr_view.extent(0), KOKKOS_LAMBDA (const int i) {
      firsts(i) = binary_search_first(i, arr_view);
      lasts(i) = binary_search_last(i, arr_view);
    });

    auto h_firsts = Kokkos::create_mirror_view(firsts);
    auto h_lasts = Kokkos::create_mirror_view(lasts);
    Kokkos::deep_copy(h_firsts, firsts);
    Kokkos::deep_copy(h_lasts, lasts);

    for (int i=0; i<arr_view.extent(0); ++i) {
      logger.debug("{} found at (first, last) = ({}, {})", i, h_firsts(i), h_lasts(i));
      REQUIRE( h_lasts(i) >= h_firsts(i));
      if (i==3) {
        REQUIRE(firsts(i) == 3);
        REQUIRE(lasts(i) == 5);
      }
      else if (i==4 or i==5) {
        REQUIRE(firsts(i) == NULL_IDX);
        REQUIRE(lasts(i) == NULL_IDX);
      }
      else {
        REQUIRE(firsts(i) == i);
        REQUIRE(lasts(i) == i);
      }
    }


    logger.info("binary search tests pass.");
  }
}
