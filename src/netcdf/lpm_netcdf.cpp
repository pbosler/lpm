#include "netcdf/lpm_netcdf.hpp"

namespace Lpm {

bool has_nc_file_extension(const std::string& filename) {
  const auto dot_pos = filename.find_last_of('.');
  const bool nc = filename.substr(dot_pos+1) == "nc";
  return nc;
}

std::string nc_handle_errcode(const int& ec, const std::string& file, const std::string& fn,
  const int& line) {
  std::ostringstream ss;
  ss << "netCDF error in file: " << file << ", function: " << fn <<", at line: "<<line << "\n";
  ss << "\terror " << ec << " decodes to: ";
  switch (ec) {
    case (NC_NOERR): {
      // no error
      return "";
    }
    case (NC_EEXIST): {
      ss << "File exists, and overwrite is not permitted.";
      break;
    }
    case (NC_EPERM): {
      ss << "cannot create file; a permission issue, or trying to write to "
            "read-only file.";
      break;
    }
    case (NC_ENOMEM): {
      ss << "out of memory.";
      break;
    }
    case (NC_ENFILE): {
      ss << "too many netcdf files open.";
      break;
    }
    case (NC_EHDFERR): {
      ss << "unknown hdf5 error.";
      break;
    }
    case (NC_EFILEMETA): {
      ss << "metadata write error.";
      break;
    }
    case (NC_EDISKLESS): {
      ss << "error creating file in memory.";
      break;
    }
    case (NC_EINVAL): {
      ss << "more than one fill value defined, or trying to set global "
            "_FillValue, or invalid input.";
      break;
    }
    case (NC_ENOTVAR): {
      ss << "could not locate variable id.";
      break;
    }
    case (NC_EBADTYPE): {
      ss << "fill value and var must have same type.";
      break;
    }
    case (NC_ELATEFILL): {
      ss << "Fill values must be written while file is still in 'define' mode.";
      break;
    }
    case (NC_EMAXNAME): {
      ss << "name is too long.";
      break;
    }
    case (NC_EDIMSIZE): {
      ss << "invalid dim size.";
      break;
    }
    case (NC_ENOTINDEFINE): {
      ss << "netcdf not in define mode.";
      break;
    }
    case (NC_EUNLIMIT): {
      ss << "NC_UNLIMITED is already used.";
      break;
    }
    case (NC_EMAXDIMS): {
      ss << "ndims > NC_MAX_DIMS";
      break;
    }
    case (NC_ENAMEINUSE): {
      ss << "name already used.";
      break;
    }
    case (NC_EBADNAME): {
      ss << "name breaks NetCDF naming rules.";
      break;
    }
    default: {
      ss << "unknown netcdf error; ec = " << ec;
    }
  }
  LPM_REQUIRE_MSG(false, ss.str());
}


}
