#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "LpmConfig.h"
#include "lpm_constants.hpp"
#include "lpm_geometry.hpp"
#include "lpm_vorticity_gallery.hpp"
#include "util/lpm_floating_point.hpp"

using namespace Lpm;

TEST_CASE("jm86_forcing", "") {
  const JM86Forcing forcing;

  SECTION("strength is 0.3 x planetary vorticity at the pole") {
    const Real omega = 2 * constants::PI;
    CHECK(FloatingPoint<Real>::equiv(JM86Forcing::F0, 0.3 * 2 * omega));
  }

  SECTION("time envelope") {
    CHECK(forcing.forcing_a(0) == 0);
    CHECK(FloatingPoint<Real>::equiv(forcing.forcing_a(JM86Forcing::tfull), 1));
    CHECK(FloatingPoint<Real>::equiv(forcing.forcing_a(0.5 * (JM86Forcing::tfull + JM86Forcing::tstar)), 1));
    CHECK(FloatingPoint<Real>::equiv(forcing.forcing_a(JM86Forcing::tstar), 1));
    CHECK(FloatingPoint<Real>::zero(forcing.forcing_a(JM86Forcing::tend)));
    CHECK(forcing.forcing_a(JM86Forcing::tend + 1) == 0);

    const Real h = 1e-6;
    for (const Real t : {0.7, 2.0, 3.5, 6.0, 11.5, 13.0, 14.5}) {
      const Real fd = (forcing.forcing_a(t + h) - forcing.forcing_a(t - h)) / (2 * h);
      CHECK(std::abs(fd - forcing.forcing_aprime(t)) < 1e-6);
    }
  }

  SECTION("latitude profile") {
    CHECK(forcing.forcing_b(-0.3) == 0);
    CHECK(FloatingPoint<Real>::zero(forcing.forcing_b(0)));
    // maximum value 1 at theta = b0
    CHECK(FloatingPoint<Real>::equiv(forcing.forcing_b(JM86Forcing::b0), 1));
    CHECK(FloatingPoint<Real>::zero(forcing.forcing_bprime(JM86Forcing::b0)));

    const Real h = 1e-6;
    for (const Real th : {0.2, 0.6, 0.9, 1.2, 1.4}) {
      const Real fd = (forcing.forcing_b(th + h) - forcing.forcing_b(th - h)) / (2 * h);
      CHECK(std::abs(fd - forcing.forcing_bprime(th)) < 1e-6);
    }
  }

  SECTION("forcing value and material derivative") {
    const Real theta = 1.0;
    const Real lambda = 0.4;
    const Real xyz[3] = {std::cos(theta) * std::cos(lambda), std::cos(theta) * std::sin(lambda), std::sin(theta)};
    const Real t = 2.0;
    const Real expected = JM86Forcing::F0 * forcing.forcing_a(t) * forcing.forcing_b(theta) * std::cos(lambda);
    CHECK(FloatingPoint<Real>::equiv(forcing(xyz, t), expected));

    // zero velocity: material derivative reduces to the partial time derivative
    const Real zero_vel[3] = {0, 0, 0};
    const Real dt_expected =
        JM86Forcing::F0 * forcing.forcing_aprime(t) * forcing.forcing_b(theta) * std::cos(lambda);
    CHECK(FloatingPoint<Real>::equiv(forcing.derivative(xyz, zero_vel, t), dt_expected));

    // pure zonal flow: u . grad F = (u/cos(theta)) dF/dlambda
    const Real u = 0.7;
    const Real vel[3] = {-std::sin(lambda) * u, std::cos(lambda) * u, 0};
    const Real zonal_term = -JM86Forcing::F0 * forcing.forcing_a(t) * forcing.forcing_b(theta) *
                            std::sin(lambda) * u / std::cos(theta);
    CHECK(std::abs(forcing.derivative(xyz, vel, t) - (dt_expected + zonal_term)) < 1e-10);
  }
}
