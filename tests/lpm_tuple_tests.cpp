#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <string>
#include <typeinfo>

#include "LpmConfig.h"
#include "lpm_comm.hpp"
#include "lpm_constants.hpp"
#include "lpm_logger.hpp"
#include "util/lpm_floating_point.hpp"
#include "util/lpm_string_util.hpp"
#include "util/lpm_tuple.hpp"

using namespace Lpm;

using Catch::Approx;

TEST_CASE("tuple unit tests", "") {
  SECTION("basic functions") {
    Kokkos::Tuple<Real, 1> real_tuple0;

    REQUIRE(FloatingPoint<Real>::zero(real_tuple0[0]));

    constexpr Real one = 1.0;
    Kokkos::Tuple<Real, 2> real_tuple1(one);
    for (int i = 0; i < 2; ++i) {
      REQUIRE(FloatingPoint<Real>::equiv(real_tuple1[i], 1));
    }

    Kokkos::Tuple<Real, 2> real_tuple2(1.0, 2.0);
    REQUIRE(FloatingPoint<Real>::equiv(real_tuple2[0], 1));
    REQUIRE(FloatingPoint<Real>::equiv(real_tuple2[1], 2));

    Kokkos::Tuple<Real, 3> real_tuple3(1.0, 2.0, 3.0);
    REQUIRE(FloatingPoint<Real>::equiv(real_tuple3[0], 1));
    REQUIRE(FloatingPoint<Real>::equiv(real_tuple3[1], 2));
    REQUIRE(FloatingPoint<Real>::equiv(real_tuple3[2], 3));
  }
  SECTION("reduction kernels") {
    using value_type = Kokkos::Tuple<Real, 12>;
    constexpr Int N  = 10;
    Kokkos::Tuple<Real, 12> sum;
    Kokkos::parallel_reduce(
        N,
        KOKKOS_LAMBDA(const int i, value_type& s) {
          value_type increment(i);
          s += increment;
        },
        sum);
    for (int i = 0; i < 12; ++i) {
      REQUIRE(sum[i] == (N - 1) * N / 2);
    }
  }
}
