#include "lpm_netcdf_writer.hpp"

namespace Lpm {

void NcWriter::open() {
  int retval = nc_create(fname.c_str(), NC_NETCDF4, &ncid);
  CHECK_NCERR(retval);
  text_att_type att0 = std::make_pair("LPM", "Lagrangian Particle Methods");
  define_file_attribute(att0);
}

void NcWriter::define_file_attribute(const text_att_type& att) const {
  const int att_len = att.second.size();
  int retval = nc_put_att_text(ncid, NC_GLOBAL, att.first.c_str(), att_len, att.second.c_str());
  CHECK_NCERR(retval);
}

void NcWriter::define_time_dim(const std::string& time_units) {
  LPM_ASSERT(ncid != NC_EBADID);
  time_dimid = ndims++;
  int retval = nc_def_dim(ncid, "time", NC_UNLIMITED, &time_dimid);
  CHECK_NCERR(retval);
  name_dimid_map.emplace("time", time_dimid);
  LPM_ASSERT(ndims == name_dimid_map.size());

  int varid = NC_EBADID;
  retval = nc_def_var(ncid, "time", nc_real_type::value, 1, &time_dimid, &varid);
  CHECK_NCERR(retval);
  const auto unit_str = (time_units.empty() ? "time_units" : time_units);
  retval = nc_put_att_text(ncid, varid, "units", unit_str.size(), unit_str.c_str());
  CHECK_NCERR(retval);
  name_varid_map.emplace("time", varid);
  ++nvars;
  LPM_ASSERT(nvars == name_varid_map.size());

  add_time_value(0);
}

void NcWriter::add_time_value(const Real t) const {
  const int varid = name_varid_map.at("time");
  size_t next_time_idx = 0;
  int retval = nc_inq_dimlen(ncid, time_dimid, &next_time_idx);
  CHECK_NCERR(retval);
  retval = nc_put_var1(ncid, varid, &next_time_idx, &t);
  CHECK_NCERR(retval);
}

} // namespace Lpm
