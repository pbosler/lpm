#include "netcdf/lpm_netcdf.hpp"

namespace Lpm {

template <typename Geo>
void NcWriter<Geo>::open() {
  int retval = nc_create(fname.c_str(), NC_NETCDF4 | NC_CLOBBER, &ncid);
  CHECK_NCERR(retval);

  text_att_type att = std::make_pair("LPM", "Lagrangian Particle Methods");
  define_file_attribute(att);

  att = std::make_pair("LPM_version", std::string(version()));
  define_file_attribute(att);

  att = std::make_pair("LPM_revision", std::string(revision()));
  define_file_attribute(att);

  att = std::make_pair("LPM_has_uncommitted_changes", (has_uncomitted_changes() ? "true" : "false"));
  define_file_attribute(att);
}

template <typename Geo>
void NcWriter<Geo>::define_file_attribute(const text_att_type& att_pair) const {
  const int att_len = att_pair.second.size();
  int retval = nc_put_att_text(ncid, NC_GLOBAL, att_pair.first.c_str(), att_len,
    att_pair.second.c_str());
  CHECK_NCERR(retval);
}

template <typename Geo>
void NcWriter<Geo>::define_time_dim() {
  LPM_ASSERT(ncid != NC_EBADID);
  LPM_REQUIRE_MSG(time_dimid == NC_EBADID, "time dimension already defined.");
  int retval = nc_def_dim(ncid, "time", NC_UNLIMITED, &time_dimid);
  CHECK_NCERR(retval);
  n_nc_dims++;
}

template <typename Geo>
void NcWriter<Geo>::define_particles_dim(const Index np) {
  LPM_ASSERT(ncid != NC_EBADID);
  LPM_REQUIRE(particles_dimid != NC_EBADID);
  int retval = nc_def_dim(ncid, "n_particles", np, &particles_dimid);
  CHECK_NCERR(retval);
  n_nc_dims++;
}

template <typename Geo>
void NcWriter<Geo>::define_coord_dim() {
  LPM_ASSERT(ncid != NC_EBADID);
  LPM_REQUIRE(coord_dimid == NC_EBADID);
  int retval = nc_def_dim(ncid, "coord", Geo::ndim);
  CHECK_NCERR(retval);
  n_nc_dims++;
}

template <typename Geo> template <typename FaceType>
void NcWriter<Geo>::define_faces(const Faces<FaceType, Geo>& faces) {
  LPM_ASSERT(ncid != NC_EBADID);
  LPM_REQUIRE_MSG(faces_dimid == NC_EBADID, "faces dimension already defined.");
  int retval = nc_def_dim(ncid, "faces", faces.nh(), &faces_dimid);
  CHECK_NCERR(retval);
  n_nc_dims++;
}

template <typename Geo> template <FieldLocation FL>
void NcWriter<Geo>::define_scalar_field(const ScalarField<FL>& s) {
  LPM_ASSERT(ncid != NC_EBADID);
  LPM_ASSERT(time_dimid != NC_EBADID);
  int m_ndims = 2;
  int dimids[2];
  switch (FL) {
    case( ParticleField ) : {
      LPM_REQUIRE(particles_dimid != NC_EBADID);
      dimids  = {time_dimid, particles_dimid};
      break;
    }
    case( VertexField ) : {
      LPM_REQUIRE(vertices_dimid != NC_EBADID);
      dimids = {time_dimid, vertices_dimid};
      break;
    }
    case( EdgeField ) : {
      LPM_REQUIRE(edges_dimid != NC_EBADID);
      dimids = {time_dimid, edges_dimid};
      break;
    }
    case( FaceField ) : {
      LPM_REQUIRE(faces_dimid != NC_EBADID);
      dimids = {time_dimid, faces_dimid};
      break;
    }
  }
  int varid = NC_EBADID;
  int retval = nc_def_var(ncid, s.name.c_str(), nc_real_type::value, m_ndims, dimids, &varid);
  CHECK_NCERR(retval);
  name_varid_map.emplace(s.name, varid);
  for (auto& md : s.metadata) {
    retval = nc_put_att_text(ncid, varid, md.first.c_str(), md.second.size(), md.second.c_str());
    CHECK_NCERR(retval);
  }
}

template <typename Geo>
void NcWriter<Geo>::define_edges(const Edges& edges) {
  LPM_ASSERT(ncid != NC_EBADID);
  LPM_REQUIRE_MSG(edges_dimid == NC_EBADID, "edges dimension already defined.");
  LPM_REQUIRE(two_dimid == NC_EBADID);
  int retval = nc_def_dim(ncid, "edges", edges.nh(), &edges_dimid);
  CHECK_NCERR(retval);
  n_nc_dims++;

  retval = nc_def_dim(ncid, "two", 2, &two_dimid);
  CHECK_NCERR(retval);
  n_nc_dims++;

  const auto nedges = edges.nh();
  const auto nmaxedges = edges.n_max();
  const auto nleaves = edges._hn_leaves();

  int origs_varid = NC_EBADID;
  int dests_varid = NC_EBADID;
  int lefts_varid = NC_EBADID;
  int rights_varid = NC_EBADID;
  int kids_varid = NC_EBADID;
  int parents_varid = NC_EBADID;
  retval = nc_def_var(ncid, "edges.origs", nc_index_type::value, 1, &edges_dimid, &origs_varid);
  CHECK_NCERR(retval);
  name_varid_map.emplace("edges.origs", origs_varid);
  retval = nc_put_att(ncid, origs_varid, "edges.n_max", nc_index_type::value, 1, &nmaxedges);
  CHECK_NCERR(retval);
  retval = nc_put_att(ncid, origs_varid, "edges.n_leaves", nc_index_type::value, 1, &nleaves);
  CHECK_NCERR(retval);
  retval = nc_def_var(ncid, "edges.dests", nc_index_type::value, 1, &edges_dimid, &dests_varid);
  CHECK_NCERR(retval);
  name_varid_map.emplace("edges.dests", dests_varid);
  retval = nc_def_var(ncid, "edges.lefts", nc_index_type::value, 1, &edges_dimid, &lefts_varid);
  CHECK_NCERR(retval);
  name_varid_map.emplace("edges.lefts", lefts_varid);
  retval = nc_def_var(ncid, "edges.rights", nc_index_type::value, 1, &edges_dimid, &rights_varid);
  CHECK_NCERR(retval);
  name_varid_map.emplace("edges.rights", rights_varid);
  const int kid_dims[2] = {edges_dimid, two_dimid};
  retval = nc_def_var(ncid, "edges.kids", nc_index_type::value, 2, kid_dims, &kids_varid);
  CHECK_NCERR(retval);
  name_varid_map.emplace("edges.kids", kids_varid);
  retval = nc_def_var(ncid, "edges.parent", nc_index_type::value, 1, &edges_dimid, &parents_varid);
  CHECK_NCERR(retval);
  name_varid_map.emplace("edges.parent", parents_varid);

  const size_t start = 0;
  const size_t count = nedges;
  retval = nc_put_vara(ncid, origs_varid, &start, &count, edges._ho.data());
  CHECK_NCERR(retval);
  retval = nc_put_vara(ncid, dests_varid, &start, &count, edges._hd.data());
  CHECK_NCERR(retval);
  retval = nc_put_vara(ncid, lefts_varid, &start, &count, edges._hl.data());
  CHECK_NCERR(retval);
  retval = nc_put_vara(ncid, rights_varid, &start, &count, edges._hr.data());
  CHECK_NCERR(retval);
  retval = nc_put_vara(ncid, parents_varid, &start, &count, edges._hp.data());

  for (size_t i=0; i<nedges; ++i) {
    for (size_t j=0; j<2; ++j) {
      const size_t idx[2] = {i,j};
      retval = nc_put_var1(ncid, kids_varid, idx, edges.kid_host(i,j));
      CHECK_NCERR(retval);
    }
  }

}

template <typename Geo>
void NcWriter<Geo>::define_vertices(const Vertices<Coords<Geo>>& vertices) {
  LPM_ASSERT(ncid != NC_EBADID);
  LPM_REQUIRE_MSG(vertices_dimid == NC_EBADID, "vertices dimension already defined.");
  const Index nverts = vertices.nh();
  int retval = nc_def_dim(ncid, "vertices", vertices.nh(), &vertices_dimid);
  CHECK_NCERR(retval);
  n_nc_dims++;
  const auto h_inds = vertices.host_crd_inds();
  {
    int varid = NC_EBADID;
    retval = nc_def_var(ncid, "vertices.crd_inds", nc_index_type::value, 1, &vertices_dimid, &varid);
    CHECK_NCERR(retval);
    name_varid_map.emplace("vertices.crd_inds", varid);
    size_t start=0;
    size_t count=nverts;
    retval = nc_put_vara(ncid, varid, &start, &count, h_inds.data());
    CHECK_NCERR(retval);
    const auto nmaxverts = vertices.n_max();
    retval = nc_put_att(ncid, varid, "n_max", nc_index_type::value, 1, &nmaxverts);
    CHECK_NCERR(retval);
  }
  {
    int varid = NC_EBADID;
    const int m_ndims = 3;
    const int dimids[3] = {time_dimid, vertices_dimid, coord_dimid};
    retval = nc_def_var(ncid, "vertices.phys_crds", nc_real_type::value, m_ndims, &dimids, &varid);
    CHECK_NCERR(retval);
    name_varid_map.emplace("vertices.phys_crds", varid);
    for (size_t i=0; i<nverts; ++i) {
      for (size_t j=0; j<Geo::ndim; ++j) {
        const size_t idx[3] = {0, i, j};
        const Real crd_val = vertices.phys_crds->get_crd_component_host(h_inds(i),j);
        retval = nc_put_var1(ncid, varid, &idx, &crd_val);
        CHECK_NCERR(retval);
      }
    }
  }
  {
    int varid = NC_EBADID;
    const int m_ndims = 3;
    const int dimids[3] = {time_dimid, vertices_dimid, coord_dimid};
    retval = nc_def_var(ncid, "vertices.lag_crds", nc_real_type::value, m_ndims, &dimids, &varid);
    CHECK_NCERR(retval);
    name_varid_map.emplace("vertices.lag_crds", varid);
    for (size_t i=0; i<nverts; ++i) {
      for (size_t j=0; j<Geo::ndim; ++j) {
        const size_t idx[3] = {0, i, j};
        const Real crd_val = vertices.lag_crds->get_crd_component_host(h_inds(i),j);
        retval = nc_put_var1(ncid, varid, &idx, &crd_val);
        CHECK_NCERR(retval);
      }
    }
  }
  {
    //TODO: If verts are dual...
  }
}

template <typename Geo> template <FieldLocation FL>
void NcWriter<Geo>::define_vector_field(const VectorField<Geo,FL>& v) {
  LPM_ASSERT(ncid != NC_EBADID);
  LPM_ASSERT(time_dimid != NC_EBADID);
  LPM_ASSERT(coord_dimid != NC_EBADID);
  int m_ndims = 3;
  int dimids[3];
  switch (FL) {
    case( ParticleField ) : {
      LPM_REQUIRE(particles_dimid != NC_EBADID);
      dimids  = {time_dimid, particles_dimid, coord_dimid};
      break;
    }
    case( VertexField ) : {
      LPM_REQUIRE(vertices_dimid != NC_EBADID);
      dimids = {time_dimid, vertices_dimid, coord_dimid};
      break;
    }
    case( EdgeField ) : {
      LPM_REQUIRE(edges_dimid != NC_EBADID);
      dimids = {time_dimid, edges_dimid, coord_dimid};
      break;
    }
    case( FaceField ) : {
      LPM_REQUIRE(faces_dimid != NC_EBADID);
      dimids = {time_dimid, faces_dimid, coord_dimid};
      break;
    }
  }
  int varid = NC_EBADID;
  int retval = nc_def_var(ncid, v.name.c_str(), nc_real_type::value, m_ndims, dimids, &varid);
  CHECK_NCERR(retval);
  name_varid_map.emplace(v.name, varid);
  for (auto& md : v.metadata) {
    retval = nc_put_att_text(ncid, varid, md.first.c_str(), md.second.size(), md.second.c_str());
    CHECK_NCERR(retval);
  }
}

template <typename Geo>
void NcWriter<Geo>::define_single_real_var(const std::string& name,
  const ekat::units::Units& units, const Real val, const std::vector<text_att_type> metadata) {
  LPM_ASSERT(ncid != NC_EBADID);

  int* ignore_me;
  int varid = NC_EBADID;
  int retval = nc_def_var(ncid, name.c_str(), nc_real_type::value, 0, ignore_me, &varid);
  CHECK_NCERR(retval);
  name_varid_map.emplace(name, varid);
  const auto unitstr = ekat::units::to_string(units);
  retval = nc_put_att_text(ncid, varid, "units", unitstr.size(), unitstr.c_str());
  for (auto& md : metadata) {
    retval = nc_put_att_text(ncid, varid, md.first.c_str(), md.second.size(), md.second.c_str());
    CHECK_NCERR(retval);
  }
  retval = nc_put_var(ncid, varid, &val);
}

template <typename Geo>
void NcWriter<Geo>::close() {
  int retval = nc_close(ncid);
  CHECK_NCERR(retval);
}

std::string nc_handle_errcode(const int& ec) {
  std::ostringstream ss;
  ss << "netCDF error: ";
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

template <typename Geo>
Index NcWriter<Geo>::n_timesteps() const {
  size_t nsteps;
  int retval = nc_inq_dimlen(ncid, time_dimid, &nsteps);
  CHECK_NCERR(retval);
  return Index(nsteps);
}

template <typename Geo>
Index NcWriter<Geo>::n_particles() const {
  size_t np;
  int retval = nc_inq_dimlen(ncid, particles_dimid, &np);
  CHECK_NCERR(retval);
  return Index(np);
}

template <typename Geo>
Index NcWriter<Geo>::n_vertices() const {
  size_t nverts;
  int retval = nc_inq_dimlen(ncid, vertices_dimid, &nverts);
  if (retval == NC_EBADDIM) {
    nverts = 0;
  }
  else {
    CHECK_NCERR(retval);
  }
  return Index(nverts);
}

template <typename Geo>
Index NcWriter<Geo>::n_edges() const {
  size_t nedges;
  int retval = nc_inq_dimlen(ncid, edges_dimid, &nedges);
  if (retval == NC_EBADDIM) {
    nedges = 0;
  }
  else {
    CHECK_NCERR(retval);
  }
  return Index(nedges);
}

template <typename Geo>
Index NcWriter<Geo>::n_faces() const {
  size_t nfaces;
  int retval = nc_inq_dimlen(ncid, faces_dimid, &nfaces);
  if (retval == NC_EBADDIM) {
    nfaces = 0;
  }
  else {
    CHECK_NCERR(retval);
  }
  return Index(nfaces);
}

bool has_nc_file_extension(const std::string& filename) {
  const auto dot_pos = filename.find_last_of('.');
  const bool nc = filename.substr(dot_pos+1) == "nc";
  return nc;
}

// ko::View<Real**> PolyMeshReader::getVertPhysCrdView() const {
//   std::multimap<std::string,NcVar>::const_iterator var_it;
//   var_it = vars.find("phys_crds_verts");
//   const auto crd_var = var_it->second;
//   return getCrdView(crd_var);
// }
//
// ko::View<Real**> PolyMeshReader::getVertLagCrdView() const {
//   const auto var_it = vars.find("lag_crds_verts");
//   const auto crd_var = var_it->second;
//   return getCrdView(crd_var);
// }
//
// ko::View<Real**> PolyMeshReader::getFacePhysCrdView() const {
//   const auto var_it = vars.find("phys_crds_faces");
//   const auto crd_var = var_it->second;
//   return getCrdView(crd_var);
// }
//
// ko::View<Real**> PolyMeshReader::getFaceLagCrdView() const {
//   const auto var_it = vars.find("lag_crds_faces");
//   const auto crd_var = var_it->second;
//   return getCrdView(crd_var);
// }
//
// ko::View<Real**> NcReader::getCrdView(const NcVar& crd_var) const {
//   const Index ncrds = crd_var.getDim(0).getSize();
//   const Index ndim = crd_var.getDim(1).getSize();
//
//   ko::View<Real**> result("crds", ncrds, ndim);
//   auto hcrds = ko::create_mirror_view(result);
//
//   std::vector<size_t> ind(2,0), count(2);
//   count[0] = 1;
//   count[1] = ndim;
//   for (Index i=0; i<ncrds; ++i) {
//     ind[0] = i;
//     Real crdvals[3];
//     crd_var.getVar(ind, count, &crdvals[0]);
//     for (Short j=0; j<ndim; ++j) {
//       hcrds(i,j) = crdvals[j];
//     }
//   }
//   ko::deep_copy(result, hcrds);
//   return result;
// }
//
// Index PolyMeshReader::nEdges() const {
//   const auto dim_it = dims.find("nedges");
//   return dim_it->second.getSize();
// }
//
// Index PolyMeshReader::nFaces() const {
//   const auto dim_it = dims.find("nfaces");
//   return dim_it->second.getSize();
// }
//
// void NcReader::fill_host_index_view(host_index_view& hv,
//   const netCDF::NcVar& ind_var) const {
//   std::vector<size_t> ind(1,0);
//   std::vector<size_t> count(1,hv.extent(0));
//   ind_var.getVar(ind, count, hv.data());
// }
//
// void NcReader::fill_host_scalar_view(host_scalar_view& hv,
//   const netCDF::NcVar& fvar) const {
//   std::vector<size_t> ind(1,0);
//   std::vector<size_t> count(1,hv.extent(0));
//   fvar.getVar(ind, count, hv.data());
// }
//
// void NcReader::fill_host_vector_view(host_vector_view& hv,
//   const netCDF::NcVar& fvar) const {
//   const auto ndim = dims.find("crd_dim")->second.getSize();
//   std::vector<size_t> ind(2,0);
//   const std::vector<size_t> count = {1,ndim};
//   for (Index i=0; i<hv.extent(0); ++i) {
//     ind[0] = i;
//     Real vvals[3];
//     fvar.getVar(ind, count, &vvals[0]);
//     for (Short j=0; j<ndim; ++j) {
//       hv(i,j) = vvals[j];
//     }
//   }
// }
//
// void PolyMeshReader::fill_origs(host_index_view& hv) const {
//   const auto var_it = vars.find("edge_origs");
//   fill_host_index_view(hv, var_it->second);
// }
//
// void PolyMeshReader::fill_dests(host_index_view& hv) const {
//   const auto var_it = vars.find("edge_dests");
//   fill_host_index_view(hv, var_it->second);
// }
//
// void PolyMeshReader::fill_lefts(host_index_view& hv) const {
//   const auto var_it = vars.find("edge_lefts");
//   fill_host_index_view(hv, var_it->second);
// }
//
// void PolyMeshReader::fill_rights(host_index_view& hv) const {
//   const auto var_it = vars.find("edge_rights");
//   fill_host_index_view(hv, var_it->second);
// }
//
// void PolyMeshReader::fill_edge_tree(host_index_view& hv,
//   typename ko::View<Index*[2]>::HostMirror& hk, Index& nleaves) const {
//   const auto pvar_it = vars.find("edge_parents");
//   const auto kvar_it = vars.find("edge_kids");
//   fill_host_index_view(hv, pvar_it->second);
//
//   nleaves = 0;
//   std::vector<size_t> ind(2,0);
//   const std::vector<size_t> count = {1,2};
//   for (Index i=0; i<pvar_it->second.getDim(0).getSize(); ++i) {
//     ind[0] = i;
//     Index kids[2];
//     kvar_it->second.getVar(ind, count, &kids[0]);
//     for (Short j=0; j<2; ++j) {
//       hk(i,j) = kids[j];
//     }
//     if (kids[0] == NULL_IND && kids[1] == NULL_IND) ++nleaves;
//   }
// }
//
// Int PolyMeshReader::getTreeDepth() const {
//   const auto att = ncfile->getAtt("baseTreeDepth");
//   Int result;
//   att.getValues(&result);
//   return result;
// }
//
// void PolyMeshReader::fill_facemask(host_mask_view& hv) const {
//   const auto var_it = vars.find("face_mask");
//   const std::vector<size_t> start(1,0);
//   const std::vector<size_t> count(1,hv.extent(0));
//   var_it->second.getVar(start,count,hv.data());
// }
//
// void PolyMeshReader::fill_face_connectivity(host_topo_view_tri& faceverts,
//   host_topo_view_tri& faceedges) const {
//   const auto vit = vars.find("face_verts");
//   const auto eit = vars.find("face_edges");
//   const auto nfaceverts = dims.find("nfaceverts")->second.getSize();
//
//   assert(nfaceverts == 3);
//
//   std::vector<size_t> ind(2,0);
//   const std::vector<size_t> count = {1,nfaceverts};
//   const auto vert_var = vit->second;
//   const auto edge_var = eit->second;
//   for (Index i=0; i<facevertices.extent(0); ++i) {
//     ind[0] = i;
//     Index verts[nfaceverts];
//     Index edges[nfaceverts];
//     vert_var.getVar(ind, count, &verts[0]);
//     edge_var.getVar(ind, count, &edges[0]);
//     for (Short j=0; j<nfaceverts; ++j) {
//       faceverts(i,j) = verts[j];
//       faceedges(i,j) = edges[j];
//     }
//   }
// }
//
// void PolyMeshReader::fill_face_connectivity(host_topo_view_quad& faceverts,
//   host_topo_view_quad& faceedges) const {
//   const auto vit = vars.find("face_verts");
//   const auto eit = vars.find("face_edges");
//   const auto nfaceverts = dims.find("nfaceverts")->second.getSize();
//
//   assert(nfaceverts == 4);
//
//   std::vector<size_t> ind(2,0);
//   const std::vector<size_t> count = {1,nfaceverts};
//   const auto vert_var = vit->second;
//   const auto edge_var = eit->second;
//   for (Index i=0; i<facevertices.extent(0); ++i) {
//     ind[0] = i;
//     Index verts[nfaceverts];
//     Index edges[nfaceverts];
//     vert_var.getVar(ind, count, &verts[0]);
//     edge_var.getVar(ind, count, &edges[0]);
//     for (Short j=0; j<nfaceverts; ++j) {
//       faceverts(i,j) = verts[j];
//       faceedges(i,j) = edges[j];
//     }
//   }
// }
//
// void PolyMeshReader::fill_face_centers(host_index_view& hv) const {
//   const auto var_it = vars.find("face_centers");
//   fill_host_index_view(hv, var_it->second);
// }
//
// void PolyMeshReader::fill_face_levels(host_index_view& hv) const {
//   const auto var_it = vars.find("face_tree_level");
//   fill_host_index_view(hv, var_it->second);
// }
//
// void PolyMeshReader::fill_face_tree(host_index_view& hp,
//   typename ko::View<Index*[4]>::HostMirror& hk, Index& nleaves) const {
//   const auto pvar_it = vars.find("face_parents");
//   const auto kvar_it = vars.find("face_kids");
//   fill_host_index_view(hp, pvar_it->second);
//
//   nleaves = 0;
//   std::vector<size_t> ind(2,0);
//   const std::vector<size_t> count = {1,4};
//   const auto kvar = kvar_it->second;
//   for (Index i=0; i<hp.extent(0); ++i) {
//     ind[0] = i;
//     Index kids[4];
//     kvar.getVar(ind, count, &kids[0]);
//     Short kid_counter = 0;
//     for (Short j=0; j<4; ++j) {
//       hk(i,j) = kids[j];
//       if (kids[j] != NULL_IND) ++kid_counter;
//     }
//     if (kid_counter == 0) ++nleaves;
//   }
// }
//
// void PolyMeshReader::fill_face_area(typename scalar_view_type::HostMirror& hv) const {
//   const auto it = vars.find("face_area");
//   fill_host_scalar_view(hv, it->second);
// }

}
