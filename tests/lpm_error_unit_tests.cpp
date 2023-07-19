#include "LpmConfig.h"
#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
#include "lpm_error.hpp"
#include "lpm_error_impl.hpp"
#include "Kokkos_Core.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using namespace Lpm;

using Catch::Approx;

TEST_CASE("error norms", "") {
  Comm comm;
  Logger<> logger("lpm_error_unit_tests", Log::level::debug, comm);

  const Int npts = 300;
  scalar_view_type ones("ones", npts);
  scalar_view_type appx_ones("appx_ones", npts);
  Kokkos::deep_copy(ones, 1);
  Kokkos::deep_copy(appx_ones, 1.001);

  scalar_view_type ones_error("ones_error", npts);

  ErrNorms ones_enorms(ones_error, appx_ones, ones, ones);

  logger.info(ones_enorms.info_string());

  REQUIRE(ones_enorms.l1 == Approx(0.001));
  REQUIRE(ones_enorms.l2 == Approx(0.001));
  REQUIRE(ones_enorms.linf == Approx(0.001));
}
