#include "LpmConfig.h"
#include "util/lpm_math.hpp"
#include "util/lpm_floating_point.hpp"
#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
#include "catch.hpp"
#include <typeinfo>
#include <fstream>
#include <string>

using namespace Lpm;

TEST_CASE("lpm_math", "") {

  Comm comm;

  Logger<> logger("math_test_log", Log::level::info, comm);
  logger.info("atan4(0,0) = {}, expected: 0", atan4(0,0));
  REQUIRE(FloatingPoint<Real>::zero(atan4(0,0)));

  const Real a = 1;
  const Real b = -3;
  const Real c = 2;
  const Real d = 4;
  const Real det = two_by_two_determinant(a, b, c, d);
  logger.info("2x2 determinant = {}; expected: 10", det);
  REQUIRE(Lpm::FloatingPoint<Real>::equiv(det, 10));

  Real r1, r2;
  quadratic_roots(r1, r2, a, b, c);
  logger.info("quadratic root 1 = {}; expected: 2", r1);
  logger.info("quadratic root 2 = {}; expected: 1", r2);
  REQUIRE( FloatingPoint<Real>::equiv(r1, 2));
  REQUIRE( FloatingPoint<Real>::equiv(r2, 1));

#ifdef LPM_USE_BOOST
  logger.info("Checking that BesselJ0 is even");
  REQUIRE( FloatingPoint<Real>::equiv(cyl_bessel_j(0, 0.5), cyl_bessel_j(0, -0.5)));
  logger.info("Checking that BesselJ1 is odd");
  REQUIRE( FloatingPoint<Real>::equiv(cyl_bessel_j(1, 0.5), -cyl_bessel_j(1, -0.5)));

  logger.info("Checking that LegendreP(l,1) = 1 for l in {0,1,2,3,4}");
  for (int l=0; l<5; ++l) {
    REQUIRE( FloatingPoint<Real>::equiv(legendre_p(l, 1), 1));
  }
#endif
}


