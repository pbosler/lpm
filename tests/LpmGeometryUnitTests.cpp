#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmUtilities.hpp"
#include <typeinfo>
#include <fstream>
#include <string>

using namespace Lpm;

int main(int argc, char* argv[]) {
ko::initialize(argc, argv);
{
    { //PLANAR TESTS
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
    std::cout << "shape(vecs) = (" << vecs.extent(0) << ", " << vecs.extent(1) << ")" << std::endl;

    std::cout << typeid(vecs).name() << std::endl;

    // execute functions on device
    ko::parallel_for(1, KOKKOS_LAMBDA (int i) {
        scalar_result() = PlaneGeometry::triArea<ko::View<Real[2],Dev>>(r2device_a, r2device_b, r2device_c);
        if (scalar_result() != 4) error("triArea error");
        for (int j=0; j<2; ++j) {
            vecs(0,j) = r2device_a(j);
            vecs(1,j) = r2device_b(j);
            vecs(2,j) = r2device_c(j);
            vecs(3,j) = r2device_d(j);
        }
        PlaneGeometry::midpoint(result, slice(vecs, 0), slice(vecs, 1));
        if (result(0) != 1.5 || result(1) != 0.5) {
            printf("device midpoint error.\n");
            printf("vecs(0,:) = (%f,%f)\n", vecs(0,0), vecs(0,1));
            printf("vecs(1,:) = (%f,%f)\n", vecs(1,0), vecs(1,1));
            auto slice0 = slice(vecs,0);
            printf("slice(vecs,0) = (%f,%f)\n", slice0[0], slice0[1]);
            printf("midpt result = (%f,%f)\n", result(0), result(1));
            error("midpt error\n");
        }
        PlaneGeometry::barycenter(result, vecs, 3);
        if (result(0) != 2.0/3.0 || result(1) != -3) error("barycenter error\n");
        scalar_result() = PlaneGeometry::distance(slice(vecs, 0), slice(vecs,2));
        if ( std::abs(scalar_result() - 2*std::sqrt(26.0)) > ZERO_TOL) error("distance error\n");
        scalar_result() = PlaneGeometry::polygonArea(vecs, 4);
        if (std::abs(scalar_result() - 10.5) > ZERO_TOL) error("polygonArea error.");
        scalar_result() = PlaneGeometry::dot(r2device_a, r2device_b);
        });
    // copy results to host
    ko::deep_copy(host_result, result);
    prarr("barycenter", host_result.data(), 2);
    LPM_THROW_IF( (host_result(0) != 2.0/3.0 || host_result(1) != -3), "barycenter error\n");
    ko::deep_copy(host_scalar, scalar_result);
    LPM_THROW_IF( host_scalar() != 2, "dot product error.\n");
    PlaneGeometry::normalize(host_result);
    LPM_THROW_IF( PlaneGeometry::mag(host_result) != 1, "normalize error.");
    } // END PLANAR TESTS

    { // SPHERICAL TESTS

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
    std::cout << "lat(p0) = " << lat << "\n";
    const Real lat4 = SphereGeometry::latitude(p4);
    std::cout << "lat(p4) = " << lat4 << "\n";
    const Real colat4 = SphereGeometry::colatitude(p4);
    std::cout << "colat(p4) = " << colat4 << "\n";
    const Real colattest = std::abs(colat4 + lat4 - 0.5*PI);
    std::cout << "abs(colat + lat - pi/2) = " << colattest << "\n";
    LPM_THROW_IF(colattest > ZERO_TOL, "colatitude + latitude != pi/2");

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
        res() = SphereGeometry::triArea(a, b, c);
        if (std::abs(res() - PI/3.0) > ZERO_TOL) error("sphereTriArea error.");
        for (int j=0; j<3; ++j) {
            vecs(0, j) = a(j);
            vecs(1, j) = b(j);
            vecs(2, j) = c(j);
            vecs(3, j) = d(j);
        }
        SphereGeometry::midpoint(vres, slice(vecs, 0), slice(vecs,1));
        printf("sphere midpoint = (%f,%f,%f)\n", vres(0), vres(1), vres(2));
        if (std::abs(vres(0) - 0.5*std::sqrt(2.0)) > ZERO_TOL) error("sphereMidpoint error.");
        SphereGeometry::cross(vres, a, b);
        if (std::abs(vres(0)-2.0/3.0) > ZERO_TOL ||
            std::abs(vres(1)-2.0/3.0) > ZERO_TOL ||
            std::abs(vres(2)) > ZERO_TOL) error("cross product error.");
        res() = SphereGeometry::distance(a,b);
        printf("distance(a,b) = %f\n", res());
        SphereGeometry::barycenter(vres, vecs, 4);
        if (vres(0) != 1 || vres(1) != 0 || vres(2) != 0) error("sphere barycenter error.");
        res() = SphereGeometry::polygonArea(vres, vecs, 4);
        if (std::abs(res()-2*PI/3) > ZERO_TOL) error("sphere polygon error.");
    });


    } // END SPHERICAL TESTS

    { // circle tests

      const Real r2a[2] = {0.0, 1.0};
      const Real r2b[2] = {0.0, 0.5};
      const Real r2c[2] = {0.5, 0.0};
      const Real r2d[2] = {1.0, 0.0};

      Real cmid[2];
      CircularPlaneGeometry::radial_midpoint(cmid, r2a, r2d);
      printf("endpt0 = (%f, %f) has r = %f\n", r2a[0], r2a[1], CircularPlaneGeometry::mag(r2a));
      printf("endpt1 = (%f, %f) has r = %f\n", r2d[0], r2d[1], CircularPlaneGeometry::mag(r2d));
      printf("midpt = (%f, %f) has r = %f\n", cmid[0], cmid[1], CircularPlaneGeometry::mag(cmid));
      LPM_THROW_IF(!fp_equiv(CircularPlaneGeometry::mag(cmid), 1.0), "radial midpoint test failed.");

      printf("theta_b = %f\n", CircularPlaneGeometry::theta(r2b));
      LPM_THROW_IF(!fp_equiv(CircularPlaneGeometry::theta(r2b), 0.5*PI), "theta test failed.");

      printf("dtheta(b,c) = %f\n", CircularPlaneGeometry::dtheta(r2b,r2c));
      LPM_THROW_IF(!fp_equiv(CircularPlaneGeometry::dtheta(r2b,r2c), 0.5*PI), "dtheta test failed.");

      const Real qsa = CircularPlaneGeometry::quad_sector_area(r2a, r2c);
      printf("area(a,b,c,d) = %f\n", qsa);
      LPM_THROW_IF(!fp_equiv(qsa,(0.5*(PI/2)*(square(1.0)-square(0.5)))), "sector area test faild");
      LPM_THROW_IF(!fp_equiv(4*qsa + PI*square(0.5), PI),"unit disk area test failed.");

      ko::View<Real[4][2],Host> sector("sector");
      for (Short i=0;i<2;++i) {
        sector(0,i) = r2a[i];
        sector(1,i) = r2b[i];
        sector(2,i) = r2c[i];
        sector(3,i) = r2d[i];
      }
      const Real ctr[2] = {std::cos(PI/4)*3/4, std::sin(PI/4)*3/4};
      const Real pa = CircularPlaneGeometry::polygonArea(ctr, sector, 4);
      printf("circ. polygonArea = %f\n", pa);
      LPM_THROW_IF(!fp_equiv(qsa,pa), "circular polygon area test failed.");

    }
}
std::cout << "tests pass." << std::endl;
ko::finalize();
return 0;
}


