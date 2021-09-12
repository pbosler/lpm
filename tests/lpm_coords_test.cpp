#include <iostream>
#include <sstream>
#include "lpm_coords.hpp"
#include "lpm_logger.hpp"
#include "lpm_comm.hpp"
#include "util/lpm_floating_point.hpp"
#include "catch.hpp"

using namespace Lpm;

TEST_CASE("lpm coords", "") {

    Comm comm;

    Logger<> logger("coords_test", Log::level::info, comm);

    Coords<PlaneGeometry> pc4(4);
    logger.info("plane_crds_4.n_max() = {}, pc4.nh() = {}", pc4.n_max(), pc4.nh());
    REQUIRE(pc4.n_max() == 4);
    REQUIRE(pc4.nh() == 0);

    Real a[2] = {-1, 1};
    Real b[2] = {-1, 0};
    Real c[2] = {-1, -1};
    Real d[2] = {0, -1};
    pc4.insert_host(a);
    pc4.insert_host(b);
    pc4.insert_host(c);
    pc4.insert_host(d);

    pc4.update_device();

    REQUIRE(FloatingPoint<Real>::equiv(pc4.max_radius(), sqrt(2.0)));

    logger.info("pc4.n_max() = {}, pc4.nh() = {}", pc4.n_max(), pc4.nh());
    REQUIRE(pc4.nh() == 4);
    logger.info(pc4.info_string("plane:4_crd",0,true));

    Coords<PlaneGeometry> pcr(20);
    pcr.init_random(3.0);
    logger.info(pcr.info_string("plane:random_init",0,true));

    Coords<SphereGeometry> sc4(4);
    logger.info("sphere_crds_4 n_max() = {}, nh() = {}", sc4.n_max(), sc4.nh());
    REQUIRE(sc4.n_max() == 4);
    REQUIRE(sc4.nh() == 0);

    const Real p0[3] = {0.57735026918962584,  -0.57735026918962584,  0.57735026918962584};
    const Real p1[3] = {0.57735026918962584,  -0.57735026918962584,  -0.57735026918962584};
    const Real p2[3] = {0.57735026918962584,  0.57735026918962584,  -0.57735026918962584};
    const Real p3[3] = {0.57735026918962584,  0.57735026918962584, 0.57735026918962584};
    sc4.insert_host(p0);
    sc4.insert_host(p1);
    sc4.insert_host(p2);
    sc4.insert_host(p3);

    sc4.update_device();
    REQUIRE(sc4.nh() == 4);

    logger.info(sc4.info_string("sphere_crds4", 0, true));

    Coords<SphereGeometry> scr(20);
    scr.init_random();
    logger.info(scr.info_string("sph_crds_init_random", 0, true));

    MeshSeed<QuadRectSeed> seed;
    Coords<PlaneGeometry> qr(9);
    qr.init_vert_crds_from_seed(seed);
    logger.info(qr.info_string(seed.id_string()));
    Coords<PlaneGeometry> qri(6);
    qri.init_interior_crds_from_seed(seed);
    logger.info(qri.info_string(seed.id_string()));
}
