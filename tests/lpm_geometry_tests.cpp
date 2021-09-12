#include "LpmConfig.h"
#include "lpm_geometry.hpp"
#include "lpm_assert.hpp"
#include "util/lpm_floating_point.hpp"
#include "util/lpm_string_util.hpp"
#include "lpm_constants.hpp"
#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
#include "catch.hpp"
#include <typeinfo>
#include <fstream>
#include <string>

using namespace Lpm;

TEST_CASE("lpm_geometry", "") {

  Comm comm;

  Logger<> logger("geometry_test_log", Log::level::info, comm);

  SECTION("planar tests") {
    const Real r2a[2] = {1.0, 0.0};
    const Real r2b[2] = {2.0, 1.0};
    const Real r2c[2] = {-1.0, -10.0};
    const Real r2d[2] = {-0.5, -1};

    // allocate a Vec on the device
    ko::View<Real[2],Dev> r2device_a("a");
    // allocate a mirror on host
    ko::View<Real[2],Dev>::HostMirror ha = ko::create_mirror_view(r2device_a);
    // initialize on host
    ha[0] = r2a[0];
    ha[1] = r2a[1];
    // copy to device
    ko::deep_copy(r2device_a, ha);

    ko::View<Real[2],Dev> r2device_b("b");
    ko::View<Real[2],Dev>::HostMirror hb = ko::create_mirror_view(r2device_b);
    hb[0] = r2b[0];
    hb[1] = r2b[1];
    ko::deep_copy(r2device_b, hb);

    ko::View<Real[2],Dev> r2device_c("c");
    ko::View<Real[2],Dev>::HostMirror hc = ko::create_mirror_view(r2device_c);
    hc[0] = r2c[0];
    hc[1] = r2c[1];
    ko::deep_copy(r2device_c, hc);

    ko::View<Real[2],Dev> r2device_d("d");
    ko::View<Real[2],Dev>::HostMirror hd = ko::create_mirror_view(r2device_d);
    hd[0] = r2d[0];
    hd[1] = r2d[1];
    ko::deep_copy(r2device_d, hd);

    // allocate a scalar on the device and a host mirror
    ko::View<Real,Layout,Dev> scalar_result("res");
    ko::View<Real>::HostMirror host_scalar = ko::create_mirror_view(scalar_result);
    ko::View<Real[2],Dev> result("res");


    ko::View<Real[2],Dev>::HostMirror host_result = ko::create_mirror_view(result);
    ko::View<Real*[2],Dev> vecs("vecs", 4);
    logger.info("shape(vecs) = ({}, {})", vecs.extent(0), vecs.extent(1));

    logger.info("typeid(vecs).name(): {}", typeid(vecs).name());

    // execute functions on device
    ko::parallel_for(1, KOKKOS_LAMBDA (int i) {
      // test triangle area on device
      scalar_result() = PlaneGeometry::tri_area<ko::View<Real[2],Dev>>(r2device_a, r2device_b, r2device_c);
      LPM_KERNEL_REQUIRE(scalar_result() == 4);

      for (int j=0; j<2; ++j) {
          vecs(0,j) = r2device_a(j);
          vecs(1,j) = r2device_b(j);
          vecs(2,j) = r2device_c(j);
          vecs(3,j) = r2device_d(j);
      }

      LPM_KERNEL_REQUIRE(PlaneGeometry::mag(r2device_a) == 1);

      // test midpoint on device
      PlaneGeometry::midpoint(result, ko::subview(vecs, 0, ko::ALL), ko::subview(vecs, 1, ko::ALL));
      bool pass = (result(0) == 1.5) and (result(1) == 0.5);
      if (!pass) {
          printf("device midpoint error.\n");
          printf("vecs(0,:) = (%f,%f)\n", vecs(0,0), vecs(0,1));
          printf("vecs(1,:) = (%f,%f)\n", vecs(1,0), vecs(1,1));
          auto slice0 = ko::subview(vecs,0,ko::ALL);
          printf("slice(vecs,0) = (%f,%f)\n", slice0[0], slice0[1]);
          printf("midpt result = (%f,%f)\n", result(0), result(1));
      }
      LPM_KERNEL_REQUIRE(pass);

      // test barycenter
      PlaneGeometry::barycenter(result, vecs, 3);
      pass = (result(0) == 2.0/3) and (result(1) == -3);
      LPM_KERNEL_REQUIRE(pass);

      // test distance
      scalar_result() = PlaneGeometry::distance(ko::subview(vecs, 0, ko::ALL),
        ko::subview(vecs,2, ko::ALL));
      LPM_KERNEL_REQUIRE(FloatingPoint<Real>::equiv(scalar_result(), 2*sqrt(26.0)));

      // test polygon area
      scalar_result() = PlaneGeometry::polygon_area(vecs, 4);
      LPM_KERNEL_REQUIRE(FloatingPoint<Real>::equiv(scalar_result(), 10.5));

      scalar_result() = PlaneGeometry::dot(r2device_a, r2device_b);
    });
  // copy results to host
  ko::deep_copy(host_result, result);
  logger.info(sprarr("barycenter", host_result.data(), 2));
  REQUIRE( host_result(0) == 2.0/3.0 );
  REQUIRE( host_result(1) == -3);


  ko::deep_copy(host_scalar, scalar_result);
  REQUIRE( host_scalar() == 2 );
  PlaneGeometry::normalize(host_result);
  REQUIRE( PlaneGeometry::mag(host_result) == 1);
  } // END PLANAR TESTS

  SECTION("spherical tests") { // SPHERICAL TESTS

    const Real p0[3] = {0.57735026918962584,-0.57735026918962584,0.57735026918962584};
    const Real p1[3] = {0.57735026918962584,-0.57735026918962584,-0.57735026918962584};
    const Real p2[3] = {0.57735026918962584,0.57735026918962584,-0.57735026918962584};
    const Real p3[3] = {0.57735026918962584,0.57735026918962584,0.57735026918962584};
    const Real p4[3] = {0,0,1};

    ko::View<Real[3],Dev> a("a");
    ko::View<Real[3],Dev>::HostMirror ha = ko::create_mirror_view(a);
    ko::View<Real[3],Dev> b("b");
    ko::View<Real[3],Dev>::HostMirror hb = ko::create_mirror_view(b);
    ko::View<Real[3],Dev> c("c");
    ko::View<Real[3],Dev>::HostMirror hc = ko::create_mirror_view(c);
    ko::View<Real[3],Dev> d("d");
    ko::View<Real[3],Dev>::HostMirror hd = ko::create_mirror_view(d);

    const Real lat = SphereGeometry::latitude(p0);
    logger.info("lat(p0) = {}", lat);

    const Real lat4 = SphereGeometry::latitude(p4);
    logger.info("lat(p4) = {}", lat4);

    const Real colat4 = SphereGeometry::colatitude(p4);
    logger.info("colat(p4) = {}", colat4);

    const Real colattest = std::abs(colat4 + lat4 - 0.5*constants::PI);
    REQUIRE(FloatingPoint<Real>::equiv(colat4+lat4, 0.5*constants::PI));

    for (int i=0; i<3; ++i) {
        ha[i] = p0[i];
        hb[i] = p1[i];
        hc[i] = p2[i];
        hd[i] = p3[i];
    }
    ko::deep_copy(a, ha);
    ko::deep_copy(b, hb);
    ko::deep_copy(c, hc);
    ko::deep_copy(d, hd);

    ko::View<Real, Dev> res("res");
    ko::View<Real, Dev>::HostMirror hres = ko::create_mirror_view(res);
    ko::View<Real[3],Dev> vres("vres");
    ko::View<Real[3],Dev>::HostMirror hvres = ko::create_mirror_view(vres);

    ko::View<Real*[3],Dev> vecs("vecs", 4);

    ko::parallel_for(1, KOKKOS_LAMBDA (int i) {
      res() = SphereGeometry::tri_area(a, b, c);

      printf("res() = %f, pi/3 = %f, diff = %g, tol = %g\n", res(), constants::PI/3,
        abs(res() - constants::PI/3), constants::ZERO_TOL);

      LPM_KERNEL_REQUIRE(FloatingPoint<Real>::equiv(res(), constants::PI/3.0, constants::ZERO_TOL));

      for (int j=0; j<3; ++j) {
          vecs(0, j) = a(j);
          vecs(1, j) = b(j);
          vecs(2, j) = c(j);
          vecs(3, j) = d(j);
      }
      SphereGeometry::midpoint(vres, ko::subview(vecs, 0, ko::ALL),
        ko::subview(vecs, 1, ko::ALL));
      printf("sphere midpoint = (%f,%f,%f)\n", vres(0), vres(1), vres(2));

      LPM_KERNEL_REQUIRE(FloatingPoint<Real>::equiv(vres(0), 0.5*sqrt(2.0)));

      SphereGeometry::cross(vres, a, b);
      LPM_KERNEL_REQUIRE( (FloatingPoint<Real>::equiv(vres(0), 2.0/3.0, constants::ZERO_TOL) and
                           FloatingPoint<Real>::equiv(vres(1), 2.0/3.0, constants::ZERO_TOL)) and
                           FloatingPoint<Real>::zero(vres(2), constants::ZERO_TOL) );


      res() = SphereGeometry::distance(a,b);
      printf("distance(a,b) = %f\n", res());
      SphereGeometry::barycenter(vres, vecs, 4);
      LPM_KERNEL_REQUIRE( (vres(0) == 1 and vres(1) == 0) and vres(2) == 0);

      res() = SphereGeometry::polygon_area(vres, vecs, 4);

      LPM_KERNEL_REQUIRE(FloatingPoint<Real>::equiv(res(), 2*constants::PI/3, constants::ZERO_TOL));
    });


  } // END SPHERICAL TESTS

  SECTION("circle tests") { // circle tests

    const Real r2a[2] = {0.0, 1.0};
    const Real r2b[2] = {0.0, 0.5};
    const Real r2c[2] = {0.5, 0.0};
    const Real r2d[2] = {1.0, 0.0};

    Real cmid[2];
    CircularPlaneGeometry::radial_midpoint(cmid, r2a, r2d);
    printf("endpt0 = (%f, %f) has r = %f\n", r2a[0], r2a[1], CircularPlaneGeometry::mag(r2a));
    printf("endpt1 = (%f, %f) has r = %f\n", r2d[0], r2d[1], CircularPlaneGeometry::mag(r2d));
    printf("midpt = (%f, %f) has r = %f\n", cmid[0], cmid[1], CircularPlaneGeometry::mag(cmid));
    REQUIRE(FloatingPoint<Real>::equiv(CircularPlaneGeometry::mag(cmid), 1.0));

    printf("theta_b = %f\n", CircularPlaneGeometry::theta(r2b));
    REQUIRE(FloatingPoint<Real>::equiv(CircularPlaneGeometry::theta(r2b), 0.5*constants::PI));

    printf("dtheta(b,c) = %f\n", CircularPlaneGeometry::dtheta(r2b,r2c));
    REQUIRE(FloatingPoint<Real>::equiv(CircularPlaneGeometry::dtheta(r2a,r2c), 0.5*constants::PI));

    const Real qsa = CircularPlaneGeometry::quad_sector_area(r2a, r2c);
    printf("area(a,b,c,d) = %f\n", qsa);
    REQUIRE(FloatingPoint<Real>::equiv(qsa,(0.5*(constants::PI/2)*(square(1.0)-square(0.5)))));
    REQUIRE(FloatingPoint<Real>::equiv(4*qsa + constants::PI*square(0.5), constants::PI));

    ko::View<Real[4][2],Host> sector("sector");
    for (Short i=0;i<2;++i) {
      sector(0,i) = r2a[i];
      sector(1,i) = r2b[i];
      sector(2,i) = r2c[i];
      sector(3,i) = r2d[i];
    }
    const Real ctr[2] = {std::cos(constants::PI/4)*3/4, std::sin(constants::PI/4)*3/4};
    const Real pa = CircularPlaneGeometry::polygon_area(ctr, sector, 4);
    printf("circ. polygon_area = %f\n", pa);
    REQUIRE(FloatingPoint<Real>::equiv(qsa,pa));

    Real bc[2];
    CircularPlaneGeometry::barycenter(bc, sector);
    Real exbc[2] = {0.75*std::cos(constants::PI/4), 0.75*std::sin(constants::PI/4)};
    printf("bc = (%f,%f); exbc = (%f,%f)\n", bc[0], bc[1], exbc[0], exbc[1]);
    REQUIRE(FloatingPoint<Real>::equiv(bc[0], exbc[0]));
    REQUIRE( FloatingPoint<Real>::equiv(bc[1],exbc[1]));
  }
}


