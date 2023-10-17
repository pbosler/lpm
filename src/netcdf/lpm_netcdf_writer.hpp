#ifndef LPM_NETCDF_WRITER_HPP
#define LPM_NETCDF_WRITER_HPP

#include "LpmConfig.h"

#ifdef LPM_USE_NETCDF

#include "lpm_netcdf.hpp"

namespace Lpm {

class NcWriter {
  public:

  virtual ~NcWriter() {nc_close(ncid);}

  void define_file_attribute(const text_att_type& att) const;

  void define_time_dim(const std::string& time_units = std::string());

  void add_time_value(const Real t) const;

  protected:
    explicit NcWriter(const std::string& full_filename) :
      fname(full_filename),
      ncid(NC_EBADID),
      ndims(0),
      nvars(),
      time_dimid(NC_EBADID),
      name_dimid_map(),
      name_varid_map() {
        open();
      }

    std::string fname;

    int ncid;
    int ndims;
    int nvars;
    int time_dimid;

    std::map<std::string, int> name_dimid_map;
    std::map<std::string, int> name_varid_map;

    /// Open a new file and write global attributes/metadata
    void open();
};

} // namespace Lpm

#endif
#endif
