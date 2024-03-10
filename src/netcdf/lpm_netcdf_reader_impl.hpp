#ifndef LPM_NETCDF_READER_IMPL_HPP
#define LPM_NETCDF_READER_IMPL_HPP

#include "lpm_netcdf_reader.hpp"
#include "lpm_coords.hpp"
#include "lpm_field.hpp"
#include "lpm_constants.hpp"
#include "util/lpm_stl_utils.hpp"
#include "util/lpm_string_util.hpp"
#include "lpm_field.hpp"

namespace Lpm {

template <typename Geo>
UnstructuredNcReader<Geo>::UnstructuredNcReader(const std::string& full_filename) :
  NcReader(full_filename),
  n_nodes(0),
  unpacked_coords(false),
  is_lat_lon(false),
  coord_var_names()
  {
    find_coord_vars();
  }

template <typename Geo>
std::string UnstructuredNcReader<Geo>::info_string(const int tab_level) const {
  std::ostringstream ss;
  ss << NcReader::info_string(tab_level);
  auto tabstr = indent_string(tab_level);
  tabstr += "\t";
  ss << "n_nodes = " << n_nodes << "\n";
  ss << "coord_var_names : [";
  for (const auto& cv : coord_var_names) {
    ss << cv << " ";
  }
  ss << "]\n";
  return ss.str();
}

template <typename Geo>
ScalarField<ParticleField> UnstructuredNcReader<Geo>::create_scalar_field(const std::string& name) {
  if (not map_contains(name_varid_map, name) ) {
    logger.error("field {} not found in .nc file", name);
    LPM_STOP("");
  }
  auto metadata = get_field_metadata(name);
  std::string unit_str;
  if (map_contains(metadata, "units")) {
    unit_str = metadata.at("units");
    metadata.erase("units");
  }
  ScalarField<ParticleField> result(name, n_nodes, unit_str, metadata);
  const size_t start = 0;
  const size_t count = n_nodes;
  int retval = nc_get_vara(ncid, name_varid_map.at(name), &start, &count, result.hview.data());
  if (retval != NC_NOERR)  {
    const auto msg = CHECK_NCERR(retval);
    logger.error(msg);
    LPM_STOP("");
  }
  result.update_device();
  return result;
}

template <typename Geo>
ScalarField<ParticleField> UnstructuredNcReader<Geo>::create_scalar_field(const std::string& name, const int time_idx) {
  if (not map_contains(name_varid_map, name)) {
    logger.error("field {} not found in .nc file", name);
    LPM_STOP("");
  }
  LPM_REQUIRE(map_contains(name_dimid_map, "time"));
  auto metadata = get_field_metadata(name);
  std::string unit_str;
  if (map_contains(metadata, "units")) {
    unit_str = metadata.at("units");
    metadata.erase("units");
  }
  ScalarField<ParticleField> result(name, n_nodes, unit_str, metadata);

  size_t start[2];
  size_t count[2];
  const auto time_dim_id = name_dimid_map.at("time");
  if (time_dim_id == 0) {
    start[0] = time_idx;
    start[1] = 0;
    count[0] = 1;
    count[1] = n_nodes;
  }
  else {
    start[0] = 0;
    start[1] = time_idx;
    count[0] = n_nodes;
    count[1] = 1;
  }

  int retval = nc_get_vara(ncid, name_varid_map.at(name), start, count, result.hview.data());
  if (retval != NC_NOERR) {
    const auto msg = CHECK_NCERR(retval);
    logger.error(msg);
    LPM_STOP("");
  }
  result.update_device();
  return result;
}

template <typename Geo>
void UnstructuredNcReader<Geo>::find_coord_vars() {
  if constexpr (std::is_same<Geo, SphereGeometry>::value) {
    is_lat_lon = false;
    if (map_contains(name_varid_map, "lat") and map_contains(name_varid_map, "lon") ) {
      unpacked_coords = true;
      is_lat_lon = true;
      coord_var_names.push_back("lat");
      coord_var_names.push_back("lon");
    }
    else {
      if (map_contains(name_varid_map, "coord")) {
        unpacked_coords = false;
        coord_var_names.push_back("coord");
      }
      else if (map_contains(name_varid_map, "xyz")) {
        unpacked_coords = false;
        coord_var_names.push_back("xyz");
      }
      else {
        if ( (map_contains(name_varid_map, "coordx") and map_contains(name_varid_map, "coordy"))
          and map_contains(name_varid_map, "coordz") ) {
            unpacked_coords = true;
            coord_var_names.push_back("coordx");
            coord_var_names.push_back("coordy");
            coord_var_names.push_back("coordz");
        }
        else if ( (map_contains(name_varid_map, "x") and map_contains(name_varid_map, "y")) and map_contains(name_varid_map, "z") ) {
            unpacked_coords = true;
            coord_var_names.push_back("x");
            coord_var_names.push_back("y");
            coord_var_names.push_back("z");
        }
        else {
          LPM_STOP("UnstructuredNcReader error: unknown coordinate variable");
        }
      }
    }
  }
  else {
    if constexpr (std::is_same<Geo,PlaneGeometry>::value) {
      if (map_contains(name_varid_map, "xy")) {
        unpacked_coords = false;
        coord_var_names.push_back("xy");
      }
      else if (map_contains(name_varid_map, "x") and map_contains(name_varid_map, "y")) {
          unpacked_coords = true;
          coord_var_names.push_back("x");
          coord_var_names.push_back("y");
      }
    }
  }
  LPM_ASSERT( unpacked_coords == (coord_var_names.size() > 1) );
  size_t nn;
  if (map_contains(name_dimid_map, "nnodes")) {
    int retval = nc_inq_dimlen(ncid, name_dimid_map.at("nnodes"), &nn);
    if (retval != NC_NOERR) {
      const auto msg = CHECK_NCERR(retval);
      logger.error(msg);
      LPM_STOP("error loading nnodes");
    }
  }
  else if (map_contains(name_dimid_map, "n_nodes")) {
    int retval = nc_inq_dimlen(ncid, name_dimid_map.at("n_nodes"), &nn);
    if (retval != NC_NOERR) {
      const auto msg = CHECK_NCERR(retval);
      logger.error(msg);
      LPM_STOP("error loading n_nodes");
    }
  }
  else {
    logger.error("expected nnodes dimension for packed coordinates");
    LPM_STOP("");
  }
  n_nodes = Index(nn);
//   logger.debug("n_nodes = {}", n_nodes);
}

template <typename Geo>
Coords<Geo> UnstructuredNcReader<Geo>::create_coords() {
  Coords<Geo> result(n_nodes);
  result._nh() = n_nodes;
  if constexpr (std::is_same<Geo,SphereGeometry>::value) {
    if (is_lat_lon) {
      Kokkos::View<Real*> lats("lats", n_nodes);
      Kokkos::View<Real*> lons("lons", n_nodes);
      auto h_lats = Kokkos::create_mirror_view(lats);
      auto h_lons = Kokkos::create_mirror_view(lons);
      const size_t start = 0;
      const size_t count = n_nodes;
      int retval = nc_get_vara(ncid, name_varid_map.at("lat"), &start, &count, h_lats.data());
      if (retval != NC_NOERR) {
        const auto msg = CHECK_NCERR(retval);
        logger.error(msg);
        LPM_STOP("");
      }
      retval = nc_get_vara(ncid, name_varid_map.at("lon"), &start, &count, h_lons.data());
      if (retval != NC_NOERR) {
        const auto msg = CHECK_NCERR(retval);
        logger.error(msg);
        LPM_STOP("");
      }
      Kokkos::deep_copy(lats, h_lats);
      Kokkos::deep_copy(lons, h_lons);
      constexpr Real deg2rad = 1/constants::RAD2DEG;
      Kokkos::parallel_for("define_lat_lon_coords", n_nodes,
        KOKKOS_LAMBDA (const Index i) {
          auto mxyz = Kokkos::subview(result.view, i, Kokkos::ALL);
          SphereGeometry::xyz_from_lon_lat(mxyz,
             lons(i) * deg2rad, lats(i) * deg2rad);
        });
      result.update_host();
    }
    else {
      LPM_STOP("UnstructuredNcReader::create_coords error: not implemented yet");
    }
  }
  else {
    if (unpacked_coords) {
      Kokkos::View<Real*> x("x", n_nodes);
      Kokkos::View<Real*> y("y", n_nodes);
      auto h_x = Kokkos::create_mirror_view(x);
      auto h_y = Kokkos::create_mirror_view(y);
      const size_t start = 0;
      const size_t count = n_nodes;
      int retval = nc_get_vara(ncid, name_varid_map.at("x"), &start, &count, h_x.data());

      if (retval != NC_NOERR)  {
        const auto msg = CHECK_NCERR(retval);
        logger.error(msg);
        LPM_STOP("");
      }
      retval = nc_get_vara(ncid, name_varid_map.at("y"), &start, &count, h_y.data());
      if (retval != NC_NOERR)  {
        const auto msg = CHECK_NCERR(retval);
        logger.error(msg);
        LPM_STOP("");
      }

      Kokkos::deep_copy(x, h_x);
      Kokkos::deep_copy(y, h_y);
      auto crds = result.view;
      Kokkos::parallel_for(n_nodes, KOKKOS_LAMBDA (const Index i) {
        crds(i,0) = x(i);
        crds(i,1) = y(i);
      });
      result.update_host();
    }
    else {
      size_t start[2], count[2];
      start[0] = 0;
      start[1] = 0;
      count[0] = 2;
      count[1] = n_nodes;
      int retval = nc_get_vara(ncid, name_varid_map.at("xy"), start, count, result.get_host_crd_view().data());
      if (retval != NC_NOERR) {
        const auto msg = CHECK_NCERR(retval);
        logger.error(msg);
        LPM_STOP("");
      }
      result.update_device();
    }
  }
  return result;
}

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
