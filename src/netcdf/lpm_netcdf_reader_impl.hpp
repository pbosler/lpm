#ifndef LPM_NETCDF_READER_IMPL_HPP
#define LPM_NETCDF_READER_IMPL_HPP

#include "lpm_netcdf_reader.hpp"
#include "lpm_coords.hpp"
#include "util/lpm_stl_utils.hpp"
#include "util/lpm_string_util.hpp"
#include "lpm_field.hpp"

namespace Lpm {

template <typename Geo>
UnstructuredNcReader<Geo>::UnstructuredNcReader(const std::string& full_filename) :
  NcReader(full_filename),
  n_nodes(0),
  nodes_dimid(NC_EBADID),
  unpacked_coords(false),
  is_lat_lon(false),
  coord_var_name()
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
  }
}

template <typename Geo>
Coords<Geo> UnstructuredNcReader<Geo>::create_coords() const {
  Coords<Geo> result(n_nodes);
  if (is_lat_lon) {
    Kokkos::View<Real*> lats("lats", n_nodes);
    Kokkos::View<Real*> lons("lons", n_nodes);
    auto h_lats = Kokkos::create_mirror_view(lats);
    auto h_lons = Kokkos::create_mirror_view(lons);
    const size_t start = 0;
    const size_t count = n_nodes;
    int retval = nc_get_vara(ncid, name_varid_map.at("lat"), &start, &count, h_lats.data());
    CHECK_NCERR(retval);
    retval = nc_get_vara(ncid, name_varid_map.at("lon"), &start, &count, h_lons.data());
    CHECK_NCERR(retval);
    Kokkos::deep_copy(lats, h_lats);
    Kokkos::deep_copy(lons, h_lons);
    Kokkos::parallel_for("define_lat_lon_coords", n_nodes,
      KOKKOS_LAMBDA (const Index i) {
        auto mxyz = Kokkos::subview(result.view, i, Kokkos::ALL);
        SphereGeometry::xyz_from_lon_lat(mxyz, lons(i), lats(i));
      });
    result.update_host();
  }
  else {
    LPM_REQUIRE_MSG(false, "UnstructuredNcReader::create_coords error: not implemented yet");
  }
  return result;
}

// template <typename Geo>
// metadata_type UnstructuredNcReader<Geo>::get_field_metadata(const std::string& field_name) const {
//   metadata_type result;
//   constexpr int MAX_NAME_LEN = 32;
//   const int varid = name_varid_map.at(name);
//   int natts;
//   int retval = nc_inq_varnatts(ncid, varid, &natts);
//   CHECK_NCERR(retval);
//   std::vector<std::string> att_names;
//   std::vector<size_t> att_lens;
//   for (int i=0; i<natts; ++i) {
//     char name[MAX_NAME_LEN];
//     retval = nc_inq_attname(ncid, varid, i, name);
//     CHECK_NCERR(retval);
//     att_names.push_back(rstrip(std::string(name)));
//     size_t len;
//     retval = nc_inq_attlen(ncid, varid, name, &len);
//     att_lens.push_back(len);
//   }
//   for (int i=0; i<natts; ++i) {
//     att_str = (char *) malloc(att_lens[i]);
//     retval = nc_get_att_text(ncid, varid, att_names[i].c_str(), att_str);
//     result.emplace(att_names[i], std::string(att_str));
//     free(att_str);
//   }
//   return result;
// }

// template <typename Geo>
// ScalarField<ParticleField> UnstructuredNcReader<Geo>::create_scalar_field(const std::string& name) const {
//   auto var_atts = get_field_metadata(name);
//   std::string unit_str = "null_unit";
//   if (map_contains(var_atts, "units")) {
//     unit_str = var_atts.at("units");
//     var_atts.erase("units");
//   }
//   ScalarField<ParticleField> result(name, n_nodes, unit_str, var_atts);
//   const size_t start = 0;
//   const size_t count = n_nodes;
//   int retval = nc_get_vara(ncid, name_varid_map.at("name"), &start, &count, result.h_view.data());
//   CHECK_NCERR(retval);
//   result.update_device();
// }

} // namespace Lpm

#endif
