#include "netcdf/lpm_netcdf.hpp"
#ifdef LPM_USE_NETCDF
#include "util/lpm_string_util.hpp"

namespace Lpm {

bool has_nc_file_extension(const std::string& filename) {
  const auto dot_pos = filename.find_last_of('.');
  const bool nc = filename.substr(dot_pos+1) == "nc";
  return nc;
}

std::string nc_decode_error(const int& ec, const std::string& file, const std::string& fn,
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
  return ss.str();
//   LPM_REQUIRE_MSG(false, ss.str());
}



// void PolymeshReader::init_dims() {
//   vertices_dimid = name_dimid_map.at("vertices");
//   edges_dimid = name_dimid_map.at("edges");
//   faces_dimid = name_dimid_map.at("faces");
//   facekind_dimid = name_dimid_map.at("facekind");
//
//   int retval = nc_get_att(ncid, NC_GLOBAL, "vertices.n_max", &nmaxverts);
//   CHECK_NCERR(retval);
//   retval = nc_get_att(ncid, NC_GLOBAL, "edges.n_max", &nmaxedges);
//   CHECK_NCERR(retval);
//   retval = nc_get_att(ncid, NC_GLOBAL, "faces.n_max", &nmaxfaces);
//   retval = nc_get_var1(ncid, name_varid_map.at("base_tree_depth"), 0, &base_tree_depth);
//   CHECK_NCERR(retval);
// }



// std::string PolymeshReader::info_string(const int tab_level) const {
//   auto tabstr = indent_string(tab_level);
//   std::ostringstream ss;
//   ss << tabstr << "PolymeshReader info:\n";
//   tabstr += "\t";
//   ss << tabstr << "filename: " << fname << "\n";
//   ss << tabstr << "vertices dim found: " << (vertices_dimid != NC_EBADID ? "true\n" : "false\n");
//   ss << tabstr << "edges dim found: " << (edges_dimid != NC_EBADID ? "true\n" : "false\n");
//   ss << tabstr << "faces dim found: " << (faces_dimid != NC_EBADID ? "true\n" : "false\n");
//   ss << tabstr << "facekind dim found: " << (facekind_dimid != NC_EBADID ? "true\n" : "false\n");
//   ss << tabstr << "nmaxverts = " << nmaxverts << "\n";
//   ss << tabstr << "nmaxedges = " << nmaxedges << "\n";
//   ss << tabstr << "nmaxfaces = " << nmaxfaces << "\n";
//   return ss.str();
// }
//
// Index PolymeshReader::n_vertices() const {
//   LPM_ASSERT(vertices_dimid != NC_EBADID);
//   size_t nverts;
//   int retval = nc_inq_dimlen(ncid, vertices_dimid, &nverts);
//   CHECK_NCERR(retval);
//   return Index(nverts);
// }
//
// Index PolymeshReader::n_edges() const {
//   LPM_ASSERT(edges_dimid != NC_EBADID);
//   size_t nedges;
//   int retval = nc_inq_dimlen(ncid, edges_dimid, &nedges);
//   CHECK_NCERR(retval);
//   return Index(nedges);
// }
//
// Index PolymeshReader::n_faces() const {
//   LPM_ASSERT(faces_dimid != NC_EBADID);
//   size_t nfaces;
//   int retval = nc_inq_dimlen(ncid, faces_dimid, &nfaces);
//   CHECK_NCERR(retval);
//   return Index(nfaces);
// }
//

//
// void PolymeshReader::fill_edges(Edges& edges) {
//   LPM_ASSERT(edges_dimid != NC_EBADID);
//
//   const Index nedges = n_edges();
//   edges._nh() = nedges;
//   Index nleaves;
//   int retval = nc_get_att(ncid, NC_GLOBAL, "edges.n_leaves", &nleaves);
//   CHECK_NCERR(retval);
//   edges._hn_leaves() = nleaves;
//
//   const size_t start = 0;
//   const size_t count = nedges;
//   retval = nc_get_vara(ncid, name_varid_map.at("edges.origs"), &start, &count, edges._ho.data());
//   CHECK_NCERR(retval);
//   retval = nc_get_vara(ncid, name_varid_map.at("edges.dests"), &start, &count, edges._hd.data());
//   CHECK_NCERR(retval);
//   retval = nc_get_vara(ncid, name_varid_map.at("edges.lefts"), &start, &count, edges._hl.data());
//   CHECK_NCERR(retval);
//   retval = nc_get_vara(ncid, name_varid_map.at("edges.rights"), &start, &count, edges._hr.data());
//   CHECK_NCERR(retval);
//   retval = nc_get_vara(ncid, name_varid_map.at("edges.parent"), &start, &count, edges._hp.data());
//   CHECK_NCERR(retval);
//
//   for (size_t i=0; i<nedges; ++i) {
//     for (size_t k=0; k<2; ++k) {
//       size_t idx[2] = {i, k};
//       retval = nc_get_var1(ncid, name_varid_map.at("edges.kids"), idx, &edges._hk(i,k));
//       CHECK_NCERR(retval);
//     }
//   }
// }

}
#endif
