#include "lpm_netcdf_reader.hpp"
#include "lpm_netcdf_reader_impl.hpp"
#ifdef LPM_USE_NETCDF

namespace Lpm {

void NcReader::inq_dims() {
  int retval = nc_inq_ndims(ncid, &ndims);
  CHECK_NCERR(retval);

  auto dimids = std::vector<int>(ndims, NC_EBADID);
  const int include_parents = 1;
  retval = nc_inq_dimids(ncid, &ndims, &dimids[0], include_parents);
  CHECK_NCERR(retval);

  char dimname[NC_MAX_NAME];
  for (auto& id : dimids) {
    retval = nc_inq_dimname(ncid, id, dimname);
    CHECK_NCERR(retval);
    name_dimid_map.emplace(std::string(dimname), id);
  }
  time_dimid = name_dimid_map.at("time");
}

void NcReader::inq_vars() {
  int retval = nc_inq_nvars(ncid, &nvars);
  CHECK_NCERR(retval);

  auto varids = std::vector<int>(nvars, NC_EBADID);

  retval = nc_inq_varids(ncid, &nvars, &varids[0]);
  CHECK_NCERR(retval);

  char varname[NC_MAX_NAME];
  for (auto& vid : varids) {
    retval = nc_inq_varname(ncid, vid, varname);
    CHECK_NCERR(retval);
    name_varid_map.emplace(std::string(varname), vid);
  }
  time_varid = name_varid_map.at("time");
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
