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
    case (NC_EBADID): {
      ss << "bad id: specified id does not refer to an open dataset";
      break;
    }
    case (NC_EINVALCOORDS): {
      ss << "specified indices are out of range";
      break;
    }
    case (NC_ENOTATT): {
      ss << "attribute not found.";
      break;
    }
    case (NC_EMAXATTS) : {
      ss << "NC_MAX_ATTRS exceeded.";
      break;
    }
    case (NC_EBADDIM): {
      ss << "invalid dimension id or name";
      break;
    }
    case (NC_EUNLIMPOS) : {
      ss << "NC_UNLIMITED in wrong position";
      break;
    }
    case (NC_EMAXVARS) : {
      ss << "NC_MAX_VARS exceeded";
      break;
    }
    case (NC_EGLOBAL) : {
      ss << "action prohibited for NC_GLOBAL varid";
      break;
    }
    case (NC_ENOTNC) : {
      ss << "not a netcdf file";
      break;
    }
    case (NC_ENORECVARS) : {
      ss << "no record vars";
      break;
    }
    case (NC_ECHAR) : {
      ss << "attempt to convert between text and numbers";
      break;
    }
    case (NC_EEDGE) : {
      ss << "start + count exceeds dimension bound";
      break;
    }
    case (NC_ESTRIDE) : {
      ss << "illegal stride";
      break;
    }
    case (NC_ERANGE) : {
      ss << "one or more of the values are out of range by the desired type";
      break;
    }
    case (NC_EVARSIZE) : {
      ss << "one or more variables violate form constraints";
      break;
    }
    case (NC_ETRUNC) : {
      ss <<  "file likely truncated or possibly corrupted";
      break;
    }
    case (NC_EAXISTYPE): {
      ss << "unknown axis type";
      break;
    }
    case (NC_EDAP) : {
      ss << "generic DAP error";
      break;
    }
    case (NC_ECURL) : {
      ss << "generic libcurl error";
      break;
    }
    case (NC_EIO) : {
      ss << "generic io error";
      break;
    }
    case (NC_ENODATA) : {
      ss << "attempted to access variable with no data";
      break;
    }
    case (NC_EDAPSVC) : {
      ss << "DAP server error";
      break;
    }
    case (NC_EDAS) : {
      ss << "malformed or inaccessible DAS";
      break;
    }
    case (NC_EDMR) : {
      ss << "Dap4 alias";
      break;
    }
    case (NC_ENOTFOUND) : {
      ss << "no such file";
      break;
    }
    case (NC_EDATADDS) : {
      ss << "malformed or inaccessible DATADDS";
      break;
    }
    case (NC_EDAPURL) : {
      ss << "malformed DAP URL";
      break;
    }
    case (NC_EDAPCONSTRAINT): {
      ss << "malformed DAP constraint";
      break;
    }
    case (NC_ETRANSLATION) : {
      ss << "untranslatable construct";
      break;
    }
    case (NC_EACCESS) : {
      ss << "access failure";
      break;
    }
    case (NC_EAUTH) : {
      ss << "authorization failure";
      break;
    }
    case (NC_ECANTREMOVE) : {
      ss << "can't remove file";
      break;
    }
    case (NC_EINTERNAL) : {
      ss << "netcdf internal error";
      break;
    }
    case (NC_EPNETCDF) : {
      ss << "error at pnetcdf layer";
      break;
    }
    case (NC4_FIRST_ERROR) : {
      ss << "internal hdf5 error";
      break;
    }
    case (NC_ECANTREAD) : {
      ss << "can't read";
      break;
    }
    case (NC_ECANTWRITE) : {
      ss << "can't write";
      break;
    }
//     case (NC_CANTCREATE) : {
//       ss << "can't create";
//       break;
//     }
    case (NC_EDIMMETA) : {
      ss << "problem with dimension metadata";
      break;
    }
    case (NC_EATTMETA) : {
      ss << "problem with attribute metadata";
      break;
    }
    case (NC_EVARMETA) : {
      ss << "problem with variable metadata";
      break;
    }
    case (NC_ENOCOMPOUND) : {
      ss << "not a compound type";
      break;
    }
    case (NC_EATTEXISTS) : {
      ss << "attribute already exists";
      break;
    }
    case (NC_ENOTNC4) : {
      ss << "attempting netcdf-4 operation on netcdf-3 file.";
      break;
    }
    case (NC_ESTRICTNC3) : {
      ss << "attempting netcdf-4 operation on strict nc3 netcdf-4 file";
      break;
    }
    case (NC_ENOTNC3) : {
      ss << "attempting netcdf-3 operation on netcdf-4 file";
      break;
    }
    case (NC_ENOPAR) : {
      ss << "parallel operation opened for non-parallel access";
      break;
    }
    case (NC_EPARINIT) : {
      ss << "error initializing parallel access";
      break;
    }
    case (NC_EBADGRPID) : {
      ss << "bad group id";
      break;
    }
    case (NC_EBADTYPID) : {
      ss << "bad type id";
      break;
    }
    case (NC_ETYPDEFINED) : {
      ss << "type already defined";
      break;
    }
//     case (NC_EBADFIELDID) : {
//       ss << "bad field id";
//       break;
//     }
    case (NC_EBADCLASS) : {
      ss << "bad class";
      break;
    }
    case (NC_EMAPTYPE) : {
      ss << "mapped access for atomic types only";
      break;
    }
    case (NC_EDIMSCALE) : {
      ss << "HDF5 dimscale error";
      break;
    }
    case (NC_ENOGRP) : {
      ss << "no group found";
      break;
    }
    case (NC_ESTORAGE) : {
      ss << "can't specify both contiguous and chunking";
      break;
    }
    case (NC_EBADCHUNK) : {
      ss << "bad chunk size";
      break;
    }
    case (NC_ENOTBUILT) : {
      ss << "attempt to use feature that was not turned on when netCDF was configured";
      break;
    }
    case (NC_ECANTEXTEND) : {
      ss << "attempt to extend dataset during ind. I/O operation";
      break;
    }
    case (NC_EMPI) : {
      ss << "MPI operation failed.";
      break;
    }
    case (NC_EFILTER) : {
      ss << "filter operation failed";
      break;
    }
    case (NC_ERCFILE) : {
      ss << "RC file error";
      break;
    }
    case (NC_ENULLPAD) : {
      ss << "header bytes not null-padded";
      break;
    }
    case (NC_EINMEMORY) : {
      ss << "in-memory file error";
      break;
    }
    case (NC_ENOFILTER) : {
      ss << "filter not defined on variable";
      break;
    }
    case (NC_ENCZARR) : {
      ss << "error at NCZarr layer";
      break;
    }
    case (NC_ES3) : {
      ss << "generic S3 error";
      break;
    }
    case (NC_EEMPTY) : {
      ss << "attempt to read empty NCZarr key";
      break;
    }
    case (NC_EOBJECT) : {
      ss << "some object exists when it should not";
      break;
    }
    case (NC_ENOOBJECT) : {
      ss << "object not found";
      break;
    }
    case (NC_EPLUGIN) : {
      ss << "unclassified failure accessing a dynamically-allocated plugin";
      break;
    }
    default: {
      ss << "unknown netcdf error (" << ec << ")";
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
