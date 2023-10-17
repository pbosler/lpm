#include "lpm_netcdf_reader.hpp"
#include "lpm_netcdf_reader_impl.hpp"
#include "lpm_assert.hpp"
#include "util/lpm_stl_utils.hpp"
#ifdef LPM_USE_NETCDF

namespace Lpm {

NcReader::NcReader(const std::string& full_filename, const Comm& comm) :
      fname(full_filename),
      name_varid_map(),
      name_dimid_map(),
      logger("NcReader", Log::level::debug, comm) {
      int retval = nc_open(full_filename.c_str(), NC_NOWRITE, &ncid);
      if (retval != NC_NOERR) {
        const auto msg = CHECK_NCERR(retval);
        logger.error(msg);
        LPM_REQUIRE_MSG(false, "error opening netcdf file");
      }
      logger.debug("ncid = {}", ncid);
      inq_dims();
      inq_vars();
    }

void NcReader::inq_dims() {
  int retval = nc_inq_ndims(ncid, &ndims);
  if (retval != NC_NOERR) {
    const auto msg = CHECK_NCERR(retval);
    logger.error(msg);
    LPM_REQUIRE_MSG(false, "error reading netcdf dimensions");
  }
  logger.debug("found {} dims", ndims);
  auto dimids = std::vector<int>(ndims, NC_EBADID);
  const int include_parents = 1;
  retval = nc_inq_dimids(ncid, &ndims, &dimids[0], include_parents);
  if (retval != NC_NOERR) {
    const auto msg = CHECK_NCERR(retval);
    logger.error(msg);
    LPM_REQUIRE_MSG(false, "error reading netcdf dimensions");
  }

  char dimname[NC_MAX_NAME+1];
  for (auto& id : dimids) {
    retval = nc_inq_dimname(ncid, id, &dimname[0]);
    if (retval != NC_NOERR) {
      const auto msg = CHECK_NCERR(retval);
      logger.error(msg);
      LPM_REQUIRE_MSG(false, "error reading netcdf dimensions");
    }
    name_dimid_map.emplace(std::string(dimname), id);
    logger.debug("found dimension: {}", std::string(dimname));
  }
  if (map_contains(name_dimid_map, "time")) {
    time_dimid = name_dimid_map.at("time");
  }
  else {
    time_dimid = LPM_NULL_IDX;
  }
}

void NcReader::inq_vars() {
  int retval = nc_inq_nvars(ncid, &nvars);
  if (retval != NC_NOERR) {
    const auto msg = CHECK_NCERR(retval);
    logger.error(msg);
    LPM_REQUIRE_MSG(false, "error reading netcdf variables");
  }

  auto varids = std::vector<int>(nvars, NC_EBADID);
  retval = nc_inq_varids(ncid, &nvars, &varids[0]);
  if (retval != NC_NOERR) {
    const auto msg = CHECK_NCERR(retval);
    logger.error(msg);
    LPM_REQUIRE_MSG(false, "error reading netcdf variables");
  }

  char varname[NC_MAX_NAME+1];
  for (auto& vid : varids) {
    retval = nc_inq_varname(ncid, vid, varname);
    if (retval != NC_NOERR) {
      const auto msg = CHECK_NCERR(retval);
      logger.error(msg);
      LPM_REQUIRE_MSG(false, "error reading netcdf variables");
    }
    name_varid_map.emplace(std::string(varname), vid);
    logger.debug("found variable: {}", std::string(varname));
  }
  if (map_contains(name_varid_map, "time")) {
    time_varid = name_varid_map.at("time");
  }
  else {
    time_varid = LPM_NULL_IDX;
  }
}

std::string NcReader::info_string(const int tab_level) const {
  auto tabstr = indent_string(tab_level);
  std::ostringstream ss;
  ss << tabstr << "NcReader info:\n";
  tabstr += "\t";
  ss << tabstr << "filename: " << fname << "\n";
  ss << tabstr << "found dims:\n";
  for (auto& dim : name_dimid_map) {
    ss << tabstr << "\t" << dim.first << "\n";
  }
  ss << tabstr << "found vars:\n";
  for (auto& var : name_varid_map) {
    ss << tabstr << "\t" << var.first << "\n";
  }
  return ss.str();
}

Int NcReader::n_timesteps() const {
  LPM_ASSERT(time_dimid != NC_EBADID);
  size_t nsteps;
  int retval = nc_inq_dimlen(ncid, time_dimid, &nsteps);
  return Int(nsteps);
}

// ETI
template class UnstructuredNcReader<PlaneGeometry>;
template class UnstructuredNcReader<SphereGeometry>;

} // namespace Lpm
#endif
