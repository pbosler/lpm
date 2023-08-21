#ifndef LPM_NETCDF_READER_IMPL_HPP
#define LPM_NETCDF_READER_IMPL_HPP

#include "lpm_netcdf_reader.hpp"
#include "lpm_coords.hpp"
#include "util/lpm_stl_utils.hpp"

namespace Lpm {

template <typename Geo>
UnstructuredNcReader<Geo>::UnstructuredNcReader(const std::string& full_filename) :
  NcReader(full_filename),
  n_nodes(0),
  nodes_dimid(NC_EBADID),
  unpacked_coords(false),
  is_lat_lon(false),
  coord_dim_name()
  {
    nodes_dimid = name_dimid_map.at("n_nodes");
    size_t nn;
    int retval = nc_inq_dimlen(ncid, nodes_dimid, &nn);
    CHECK_NCERR(retval);
    n_nodes = Index(nn);
    find_coord_var();
  }

template <typename Geo>
std::string UnstructuredNcReader<Geo>::info_string(const int tab_level) const {
  std::ostringstream ss;
  ss << NcReader::info_string(tab_level);
  auto tabstr = indent_string(tab_level);
  tabstr += "\t";
  ss << "n_nodes = " << n_nodes << "\n";
  return ss.str();
}

template <typename Geo>
void UnstructuredNcReader<Geo>::find_coord_var() {
  if constexpr (std::is_same<Geo, SphereGeometry>::value) {
    is_lat_lon = false;
    if (map_contains(name_varid_map, "lat") and map_contains(name_varid_map, "lon") ) {
      unpacked_coords = true;
      is_lat_lon = true;
    }
    else {
      if (map_contains(name_varid_map, "coord")) {
        unpacked_coords = false;
        coord_var_name = "coord";
      }
      else if (map_contains(name_varid_map, "xyz")) {
        unpacked_coords = false;
        coord_var_name = "xyz";
      }
      else {
        if ( (map_contains(name_varid_map, "coordx") and map_contains(name_varid_map, "coordy"))
          and map_contains(name_varid_map, "coordz") ) {
            unpacked_coords = true;
            coord_var_name = "coordx";
        }
        else {
          LPM_REQUIRE_MSG(false, "UnstructuredNcReader error: unknown coordinate variable");
        }
      }
    }
  else {
    if constexpr (std::is_same<Geo, PlaneGeometry>::value) {
      LPM_REQUIRE_MSG(false, "UnstructuredNcReader error: not implemented yet.");
    }
  }
}

template <typename Geo>
Coords<Geo> UnstructuredNcReader<Geo>::create_coords() const {
  Coords<Geo> result(n_nodes);
  const auto hcrds = result.get_host_crd_view();
  for (size_t i=0; i<n_nodes; ++i) {
    auto crd_i = Kokkos::subview(hcrds, i, Kokkos::ALL);
    for (size_t j=0; j<Geo::ndim; ++j) {
      int retval = nc_get_var1(ncid,
#ifndef NDEBUG
      CHECK_NCERR(retval);
#endif
    }
  }
  return result;
}

} // namespace Lpm

#endif
