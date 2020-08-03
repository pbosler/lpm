#ifndef LPM_NETCDF_HPP
#define LPM_NETCDF_HPP

#include "LpmConfig.h"
#ifdef LPM_HAVE_NETCDF

#include "LpmDefs.hpp"
#include "LpmPolyMesh2d.hpp"
#include <netcdf>
#include <memory>
#include <map>

namespace Lpm {

class NcWriter {
  public:
    NcWriter(const std::string& filename);

    template <typename SeedType>
    void writePolymesh(const std::shared_ptr<PolyMesh2d<SeedType>>& mesh);

  protected:
    std::string fname;
    std::unique_ptr<netCDF::NcFile> ncfile;
};

}
#endif
#endif
