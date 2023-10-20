#include "lpm_netcdf_reader.hpp"
#include "lpm_netcdf_reader_impl.hpp"
#include "lpm_assert.hpp"
#include "util/lpm_stl_utils.hpp"
#include "util/lpm_string_util.hpp"
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
    LPM_STOP("error opening netcdf file");
  }
//   logger.debug("ncid = {}", ncid);
  inq_dims();
  inq_vars();
}

void NcReader::inq_dims() {
  int retval = nc_inq_ndims(ncid, &ndims);
  if (retval != NC_NOERR) {
    const auto msg = CHECK_NCERR(retval);
    logger.error(msg);
    LPM_STOP("error reading netcdf dimensions");
  }
//   logger.debug("found {} dims", ndims);
  auto dimids = std::vector<int>(ndims, NC_EBADID);
  const int include_parents = 1;
  retval = nc_inq_dimids(ncid, &ndims, &dimids[0], include_parents);
  if (retval != NC_NOERR) {
    const auto msg = CHECK_NCERR(retval);
    logger.error(msg);
    LPM_STOP("error reading netcdf dimensions");
  }

  char dimname[NC_MAX_NAME+1];
  for (auto& id : dimids) {
    retval = nc_inq_dimname(ncid, id, &dimname[0]);
    if (retval != NC_NOERR) {
      const auto msg = CHECK_NCERR(retval);
      logger.error(msg);
      LPM_STOP("error reading netcdf dimensions");
    }
    name_dimid_map.emplace(std::string(dimname), id);
//     logger.debug("found dimension: {} at id {}", std::string(dimname), id);
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
    LPM_STOP("error reading netcdf variables");
  }

  auto varids = std::vector<int>(nvars, NC_EBADID);
  retval = nc_inq_varids(ncid, &nvars, &varids[0]);
  if (retval != NC_NOERR) {
    const auto msg = CHECK_NCERR(retval);
    logger.error(msg);
    LPM_STOP("error reading netcdf variables");
  }

  char varname[NC_MAX_NAME+1];
  for (auto& vid : varids) {
    retval = nc_inq_varname(ncid, vid, varname);
    if (retval != NC_NOERR) {
      const auto msg = CHECK_NCERR(retval);
      logger.error(msg);
      LPM_STOP("error reading netcdf variables");
    }
    name_varid_map.emplace(std::string(varname), vid);
//     logger.debug("found variable: {} at varid {}", std::string(varname), vid);
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

metadata_type NcReader::get_field_metadata(const std::string& varname) {
  metadata_type result;
  constexpr int MAX_NAME_LEN = 64;
  if (not map_contains(name_varid_map, varname)) {
    logger.error("cannot find {} in name_varid_map", varname);
    LPM_STOP("variable not found");
  }
  const int varid = name_varid_map.at(varname);
  int natts;
  int retval = nc_inq_varnatts(ncid, varid, &natts);
  if (retval != NC_NOERR) {
    const auto msg = CHECK_NCERR(retval);
    logger.error(msg);
    LPM_STOP("error reading variable attributes");
  }
  std::vector<std::string> att_names;
  std::vector<size_t> att_lengths;
  for (int i=0; i<natts; ++i) {
    char name[MAX_NAME_LEN];
    retval = nc_inq_attname(ncid, varid, i, name);
    if (retval != NC_NOERR) {
      const auto msg = CHECK_NCERR(retval);
      logger.error(msg);
      LPM_STOP("error reading variable attribute " + std::string(name));
    }
    att_names.push_back(rstrip(std::string(name)));
    size_t len;
    retval = nc_inq_attlen(ncid, varid, name, &len);
    if (retval != NC_NOERR) {
      const auto msg = CHECK_NCERR(retval);
      logger.error(msg);
      LPM_STOP("error reading attribute length");
    }
    att_lengths.push_back(len);
  }
  for (int i=0; i<natts; ++i) {
    auto att_str = (char*) malloc(att_lengths[i]);
    retval = nc_get_att_text(ncid, varid, att_names[i].c_str(), att_str);
    if (retval != NC_NOERR) {
      const auto msg = CHECK_NCERR(retval);
      logger.error(msg);
      LPM_STOP("error getting text attribute");
    }
    result.emplace(att_names[i], std::string(att_str));
    free(att_str);
  }
  return result;
}


Int NcReader::n_timesteps() const {
  Int result = 0;
  if (map_contains(name_dimid_map, "time")) {
    size_t nsteps;
    int retval = nc_inq_dimlen(ncid, time_dimid, &nsteps);
    result = Int(nsteps);
  }
  return result;
}

// ETI
template class UnstructuredNcReader<PlaneGeometry>;
template class UnstructuredNcReader<SphereGeometry>;

} // namespace Lpm
#endif
