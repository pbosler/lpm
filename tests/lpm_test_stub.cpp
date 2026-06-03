#include "LpmConfig.h"
#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
#include "util/lpm_floating_point.hpp"
#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"
#endif

#include <catch2/catch_test_macros.hpp>

using namespace Lpm;

TEST_CASE("lpm stub test", "") {
  // This is a stub test case
  Comm comm;
  Logger<> logger("logger_name", Log::level::debug, comm);

  logger.info(
      "you can write info, like numbers, e.g., {}, with logger messages like "
      "this one.",
      42);

  // write test code here
  // you can test things, and crash the program, with REQUIRE statements
  REQUIRE(0 == 0);

  // a test will fail, but not crash, with a CHECK statements
  CHECK(FloatingPoint<Real>::zero(0.5 * std::numeric_limits<Real>::epsilon()));

#ifdef LPM_USE_VTK
// put vtk stuff here, if you need it.
#endif

  // if we got here, tests pass
  logger.info("tests pass.");
}
