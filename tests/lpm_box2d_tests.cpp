#include <iostream>
#include "LpmConfig.h"
#include "lpm_logger.hpp"
#include "lpm_comm.hpp"
#include "lpm_geometry.hpp"
#include "tree/lpm_box2d.hpp"
#include "tree/lpm_quadtree_lookup_tables.hpp"
#include "util/lpm_floating_point.hpp"
#include "util/lpm_tuple.hpp"
#include "catch.hpp"

using namespace Lpm;
using namespace Lpm::quadtree;

TEST_CASE("box2d", "[tree]") {
  Comm comm;

  Logger <> logger("box3d_test", Log::level::info, comm);

  const bool padding = false;
  Box2d box0(-1,1,-1,1, padding);

  SECTION("unit tests") {

    logger.info("box0.area() = {}. Expected result: 4", box0.area());
    REQUIRE(FloatingPoint<Real>::equiv(box0.area(), 4));

    logger.info("box0.aspect_ratio() = {}. Expected result: 1", box0.aspect_ratio());
    REQUIRE(box0.is_square());
    const auto box00 = box0;
    Box2d box000(box0);
    REQUIRE( box0 == box00 );
    REQUIRE( box00 == box000 );

    const Real origin[2] = {0,0};
    REQUIRE(box0.contains_pt(origin));

    Real c0[3];
    box0.centroid(c0[0], c0[1]);
    const auto c00 = box0.centroid();
    logger.info("box0.centroid() = {}. Expected result: [0,0]", c00);
    REQUIRE(FloatingPoint<Real>::zero(PlaneGeometry::square_euclidean_distance(c0, c00)));
    REQUIRE(FloatingPoint<Real>::zero(PlaneGeometry::square_euclidean_distance(c0, origin)));
    REQUIRE(box0.pt_in_neighborhood(c00) == 4);

    Kokkos::Tuple<Real,2> external_pt({2, 0});
    const auto cp = box0.closest_pt_l1(external_pt);
    const auto cpexpected = Kokkos::Tuple<Real,2>({1,0});
    logger.info("box0.closest_pt_l1([2,0]) = {}. Expected result [1,0]", cp);
    REQUIRE(cp == cpexpected);
  }

  SECTION("neighborhood/region tests") {
    Kokkos::View<Box2d> box_view("box_view");
    auto h_box_view = Kokkos::create_mirror_view(box_view);
    h_box_view() = box0;

    Kokkos::View<Int[9]> neighbors("neighbors");
    Kokkos::View<Int[4]> kid_idx("kid_idx");

    Kokkos::View<Real[9][2]> nbr_query_pts("nbr_query_pts");
    Kokkos::View<Real[4][3]> kid_query_pts("kid_query_pts");

    std::vector<std::vector<Real>> nqpts(9);
    nqpts[0] = {-2, -2};
    nqpts[1] = {-2,  0};
    nqpts[2] = {-2,  2};
    nqpts[3] = { 0, -2};
    nqpts[4] = { 0,  0};
    nqpts[5] = { 0,  2};
    nqpts[6] = { 2, -2};
    nqpts[7] = { 2,  0};
    nqpts[8] = { 2,  2};

    std::vector<std::vector<Real>> kpts(4);
    kpts[0] = {-0.5, -0.5};
    kpts[1] = {-0.5,  0.5};
    kpts[2] = { 0.5, -0.5};
    kpts[3] = { 0.5,  0.5};

    auto h_nbr_query_pts = Kokkos::create_mirror_view(nbr_query_pts);
    for (auto i=0; i<9; ++i) {
      for (auto j=0; j<2; ++j) {
        h_nbr_query_pts(i,j) = nqpts[i][j];
      }
    }
    Kokkos::deep_copy(nbr_query_pts, h_nbr_query_pts);

    auto h_kid_query_pts = Kokkos::create_mirror_view(kid_query_pts);
    for (int i=0; i<4; ++i) {
      for (int j=0; j<2; ++j) {
        h_kid_query_pts(i,j) = kpts[i][j];
      }
    }
    Kokkos::deep_copy(kid_query_pts, h_kid_query_pts);

    Kokkos::parallel_for(9, KOKKOS_LAMBDA (const int i) {
      const auto qxy = Kokkos::subview(nbr_query_pts, i, Kokkos::ALL);
      neighbors(i) = box_view().pt_in_neighborhood(qxy);
    });

    Kokkos::parallel_for(4, KOKKOS_LAMBDA (const int i) {
      const auto qxy = Kokkos::subview(kid_query_pts, i, Kokkos::ALL);
      kid_idx(i) = box_view().quadtree_child_idx(qxy);
    });

    auto hnbrs = Kokkos::create_mirror_view(neighbors);
    auto hkids = Kokkos::create_mirror_view(kid_idx);
    Kokkos::deep_copy(hnbrs, neighbors);
    Kokkos::deep_copy(hkids, kid_idx);

    for (int i=0; i<9; ++i) {
      REQUIRE(hnbrs(i) == i);
    }
    logger.info("neighbor tests pass.");


    for (int i=0; i<4; ++i) {
      REQUIRE(hkids(i) == i);
    }
    logger.info("kid index tests pass.");

    const auto kids = box0.bisect_all();
    std::vector<Box2d> kids_expected(4);
    kids_expected[0] = Box2d(-1, 0, -1, 0, false);
    kids_expected[1] = Box2d(-1, 0,  0, 1, false);
    kids_expected[2] = Box2d( 0, 1, -1, 0, false);
    kids_expected[3] = Box2d( 0, 1,  0, 1, false);
    for (int i=0; i<4; ++i) {
      REQUIRE(kids[i] == kids_expected[i]);
      REQUIRE(kids[i].contains_pt(kpts[i]));
      const auto p = parent_from_child(kids[i], i);
      REQUIRE(box0 == p);
      REQUIRE(p.contains_pt(kids[i].centroid()));
    }

    logger.info("kid box tests pass.");
  }

  SECTION("bounding box tests") {

    Kokkos::View<Real[8][2]> far_pts("far_pts");
    auto h_far_pts = Kokkos::create_mirror_view(far_pts);
    for (auto i=0; i<4; ++i) {
      h_far_pts(i,0) = ( (i&2) > 0 ? 10 : -10);
      h_far_pts(i,1) = ( (i&1) > 0 ? 10 : -10);
    }
    for (auto i=4; i<8; ++i) {
      h_far_pts(i,0) = pow(-1, i) * i;
      h_far_pts(i,1) = pow(-1, i-1) * i + (i%2);
    }
    Kokkos::deep_copy(far_pts, h_far_pts);
    Box2d bbox;
    Kokkos::parallel_reduce(far_pts.extent(0), BoundingBoxFunctor(far_pts),
      Box2dReducer<>(bbox));
    const Box2d bbox_expected(-10,10, -10,10, false);
    logger.info("bounding box : {}", bbox);
    REQUIRE(bbox == bbox_expected);

    logger.info("bounding box tests pass.");
  }

  SECTION("lookup table tests") {

    const auto plut_entries = parent_lookup_table_entries();
    Kokkos::View<ParentLUT,Host> h_parent_table("h_parent_table");
    for (int i=0; i<4; ++i) {
      for (int j=0; j<9; ++j) {
        logger.debug("table_val({}, {}, h_parent_table) = {}; plut_entries[{}] = {}",
          i, j, table_val(i, j, h_parent_table), 9*i+j, plut_entries[9*i+j]);

        if (table_val(i, j, h_parent_table) != plut_entries[9*i + j]) {
          logger.error("unexpected: table_val({},{},h_parent_table) != {}",
            i, j, plut_entries[9*i+j]);
        }
        REQUIRE(table_val(i, j, h_parent_table) == plut_entries[9*i + j]);
      }
    }
    logger.info("parent lookup table test passes.");

    const auto clut_entries = child_lookup_table_entries();
    Kokkos::View<ChildLUT,Host> h_child_table("h_child_table");
    for (int i=0; i<4; ++i) {
      for (int j=0; j<9; ++j) {
        logger.debug("table_val({}, {}, h_child_table) = {}; clut_entries[{}] = {}",
          i, j, table_val(i, j, h_child_table), 9*i+j, clut_entries[9*i+j]);
        REQUIRE(table_val(i, j, h_child_table) == clut_entries[9*i+j]);
      }
    }

    logger.info("child lookup table test passes.");
  }
}
