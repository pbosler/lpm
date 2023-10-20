#include "LpmConfig.h"
#include "lpm_comm.hpp"
#include "lpm_geometry.hpp"
#include "lpm_logger.hpp"
#include "lpm_field.hpp"
#include "lpm_field_impl.hpp"
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
  const bool dump_all = false;
  const std::string label;
  const int indent = 0;

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
  SECTION("read/write unstructured lat/lon tests") {
    std::string ll_data_file(LPM_TEST_DATA_DIR);
    ll_data_file += "/lat_lon_data.nc";

    logger.debug("opening file: {}", ll_data_file);
    UnstructuredNcReader<SphereGeometry> ll_reader(ll_data_file);
    logger.debug(ll_reader.info_string());
    const auto mdata = ll_reader.get_field_metadata("ones");
    for (const auto& md : mdata) {
      logger.info("metadata {} : {}", md.first, md.second);
    }

    Coords<SphereGeometry> sphere_crds = ll_reader.create_coords();
    logger.debug(sphere_crds.info_string(label, indent, dump_all));
    const auto ones_field = ll_reader.create_scalar_field("ones");
    logger.info(ones_field.info_string());
  }
  SECTION("read/write planar unstructured tests (packed data)") {
    std::string xy_data_file(LPM_TEST_DATA_DIR);
    xy_data_file += "/xy_packed.nc";

    UnstructuredNcReader<PlaneGeometry> xy_reader(xy_data_file);
    logger.debug(xy_reader.info_string());

    Coords<PlaneGeometry> pcrds = xy_reader.create_coords();
    logger.debug(pcrds.info_string(label, indent, dump_all));
  }
  SECTION("read/write planar unstructured tests (unpacked data)") {
    std::string xy_data_file(LPM_TEST_DATA_DIR);
    xy_data_file += "/xy_unpacked.nc";

    UnstructuredNcReader<PlaneGeometry> xy_reader(xy_data_file);
    logger.debug(xy_reader.info_string());

    Coords<PlaneGeometry> pcrds = xy_reader.create_coords();
    logger.debug(pcrds.info_string(label, indent, dump_all));
  }
}
