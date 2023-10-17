#include "LpmConfig.h"
#include "lpm_comm.hpp"
#include "lpm_geometry.hpp"
#include "lpm_logger.hpp"
#include "util/lpm_floating_point.hpp"
#ifdef LPM_USE_NETCDF
#include "netcdf/lpm_netcdf.hpp"
#include "netcdf/lpm_netcdf_reader.hpp"
#include "netcdf/lpm_netcdf_reader_impl.hpp"
#include <netcdf.h>
#endif

#include <catch2/catch_test_macros.hpp>

using namespace Lpm;

TEST_CASE("lpm netcdf test", "") {

  Comm comm;
  Logger<> logger("netcdf_test", Log::level::debug, comm);

  SECTION("basic tests") {
    const std::string bad_fname = "my_ncdata.dat";
    const std::string good_fname = "my_ncdata.nc";

    logger.debug(bad_fname);
    CHECK(not has_nc_file_extension(bad_fname));
    logger.debug(good_fname);
    CHECK(has_nc_file_extension(good_fname));

    const std::string no_err_string = CHECK_NCERR(NC_NOERR);
    logger.info("no error string: {}", no_err_string);
    CHECK(no_err_string.empty());
    const std::string err_string = CHECK_NCERR(NC_ENFILE);
    logger.info("error string: {}", err_string);
    CHECK(!err_string.empty());
  }
  SECTION("read/write tests") {
    std::string ll_data_file(LPM_TEST_DATA_DIR);
    ll_data_file += "/lat_lon_data.nc";

    logger.debug("opening file: {}", ll_data_file);
    UnstructuredNcReader<SphereGeometry> ll_reader(ll_data_file);
    logger.debug(ll_reader.info_string());
  }
}
