#include <iostream>
#include "LpmConfig.h"
#include "lpm_logger.hpp"
#include "lpm_comm.hpp"
#include "lpm_geometry.hpp"
#include "tree/lpm_box3d.hpp"
#include "util/lpm_floating_point.hpp"
#include "util/lpm_tuple.hpp"
#include "catch.hpp"

using namespace Lpm;

TEST_CASE("box3d", "[tree]") {
  Comm comm;

  Logger <> logger("box3d_test", Log::level::info, comm);

  const bool no_padding = false;
  Box3d box0(-1,1,-1,1,-1,1, no_padding);

  SECTION("unit tests") {
    logger.info("box0.volume() = {}. Expected result: 8", box0.volume());
    REQUIRE(FloatingPoint<Real>::equiv(box0.volume(), 8));

    logger.info("box0.aspect_ratio() = {}. Expected result: 1", box0.aspect_ratio());
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
    logger.info("box0.centroid() = {}. Expected result: [0,0,0]", c00);
    REQUIRE(FloatingPoint<Real>::zero(SphereGeometry::square_euclidean_distance(c0, c00)));
    REQUIRE(FloatingPoint<Real>::zero(SphereGeometry::square_euclidean_distance(c0, origin)));
    REQUIRE(box0.pt_in_neighborhood(c00) == 13);

    Kokkos::Tuple<Real,3> external_pt({2, 0, 0});
    const auto cp = box0.closest_pt_l1(external_pt);
    const auto cpexpected = Kokkos::Tuple<Real,3>({1,0,0});
    logger.info("box0.closest_pt_l1([2,0,0]) = {}. Expected result [1,0,0]", cp);
    REQUIRE(cp == cpexpected);

 }
 SECTION("neighborhood/region tests") {

    Kokkos::View<Int[27]> nbr("neighborhoods");
    Kokkos::View<Int[8]> kid_idx("kid_idx");

    Kokkos::View<Real[27][3]> nbr_query_pts("nbr_query_pts");
    Kokkos::View<Real[8][3]> kid_query_pts("kid_query_pts");

    std::vector<std::vector<Real>> nqpts(27);
    nqpts[0]  = {-2, -2, -2};
    nqpts[1]  = {-2, -2,  0};
    nqpts[2]  = {-2, -2,  2};
    nqpts[3]  = {-2,  0, -2};
    nqpts[4]  = {-2,  0,  0};
    nqpts[5]  = {-2,  0,  2};
    nqpts[6]  = {-2,  2, -2};
    nqpts[7]  = {-2,  2,  0};
    nqpts[8]  = {-2,  2,  2};
    nqpts[9]  = { 0, -2, -2};
    nqpts[10] = { 0, -2,  0};
    nqpts[11] = { 0, -2,  2};
    nqpts[12] = { 0,  0, -2};
    nqpts[13] = { 0,  0,  0};
    nqpts[14] = { 0,  0,  2};
    nqpts[15] = { 0,  2, -2};
    nqpts[16] = { 0,  2,  0};
    nqpts[17] = { 0,  2,  2};
    nqpts[18] = { 2, -2, -2};
    nqpts[19] = { 2, -2,  0};
    nqpts[20] = { 2, -2,  2};
    nqpts[21] = { 2,  0, -2};
    nqpts[22] = { 2,  0,  0};
    nqpts[23] = { 2,  0,  2};
    nqpts[24] = { 2,  2, -2};
    nqpts[25] = { 2,  2,  0};
    nqpts[26] = { 2,  2,  2};

    std::vector<std::vector<Real>> kpts(8);
    kpts[0] = {-0.5, -0.5, -0.5};
    kpts[1] = {-0.5, -0.5,  0.5};
    kpts[2] = {-0.5,  0.5, -0.5};
    kpts[3] = {-0.5,  0.5,  0.5};
    kpts[4] = { 0.5, -0.5, -0.5};
    kpts[5] = { 0.5, -0.5,  0.5};
    kpts[6] = { 0.5,  0.5, -0.5};
    kpts[7] = { 0.5,  0.5,  0.5};

    auto h_nbr_query_pts = Kokkos::create_mirror_view(nbr_query_pts);
    for (int i=0; i<27; ++i) {
      for (int j=0; j<3; ++j) {
        h_nbr_query_pts(i,j) = nqpts[i][j];
      }
    }
    Kokkos::deep_copy(nbr_query_pts, h_nbr_query_pts);
    auto h_kid_query_pts = Kokkos::create_mirror_view(kid_query_pts);
    for (int i=0; i<8; ++i) {
      for (int j=0; j<3; ++j) {
        h_kid_query_pts(i,j) = kpts[i][j];
      }
    }
    Kokkos::deep_copy(kid_query_pts, h_kid_query_pts);

    Kokkos::parallel_for(27, KOKKOS_LAMBDA (const int i) {
      const auto qxyz = Kokkos::subview(nbr_query_pts, i, Kokkos::ALL);
      nbr(i) = box0.pt_in_neighborhood(qxyz);
    });
    Kokkos::parallel_for(8, KOKKOS_LAMBDA (const int i) {
      const auto qxyz = Kokkos::subview(kid_query_pts, i, Kokkos::ALL);
      kid_idx(i) = box0.octree_child_idx(qxyz);
    });

    auto hnbrs = Kokkos::create_mirror_view(nbr);
    Kokkos::deep_copy(hnbrs, nbr);
    auto hkids = Kokkos::create_mirror_view(kid_idx);
    Kokkos::deep_copy(hkids, kid_idx);

    for (int i=0; i<27; ++i) {
      logger.debug("nqpt {} located in neighborhood {}", i, hnbrs(i));
      REQUIRE(hnbrs(i) == i);
    }
    logger.info("neighborhoods test passes.");

    for (int i=0; i<8; ++i) {
      logger.debug("kpt {} located in child {}", i, hkids(i));
      REQUIRE(hkids(i) == i);
    }
    logger.info("kids index test passes.");

    const auto kids = box0.bisect_all();
    std::vector<Box3d> kids_expected(8);
    kids_expected[0] = Box3d(-1,0, -1,0, -1,0, no_padding);
    kids_expected[1] = Box3d(-1,0, -1,0,  0,1, no_padding);
    kids_expected[2] = Box3d(-1,0,  0,1, -1,0, no_padding);
    kids_expected[3] = Box3d(-1,0,  0,1,  0,1, no_padding);
    kids_expected[4] = Box3d( 0,1, -1,0, -1,0, no_padding);
    kids_expected[5] = Box3d( 0,1, -1,0,  0,1, no_padding);
    kids_expected[6] = Box3d( 0,1,  0,1, -1,0, no_padding);
    kids_expected[7] = Box3d( 0,1,  0,1,  0,1, no_padding);

    for (int i=0; i<8; ++i) {
      REQUIRE(kids[i] == kids_expected[i]);
      REQUIRE(kids[i].contains_pt(kpts[i]));
      const auto p = parent_from_child(kids[i], i);
      REQUIRE(box0 == p);
    }
    logger.info("kids region test passes.");
  }
}
