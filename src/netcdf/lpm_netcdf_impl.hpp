#ifdef LPM_USE_NETCDF
#include "netcdf/lpm_netcdf.hpp"
#include "util/lpm_string_util.hpp"
#include <sstream>

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

  int varid = NC_EBADID;
  retval = nc_def_var(ncid, "time", nc_real_type::value, 1, &time_dimid, &varid);
  CHECK_NCERR(retval);
  const auto unit_str = ekat::units::to_string(ekat::units::s);
  retval = nc_put_att_text(ncid, varid, "units", unit_str.size(), unit_str.c_str());
  CHECK_NCERR(retval);
  name_varid_map.emplace("time", varid);
  add_time_value(0);
}

template <typename Geo>
void NcWriter<Geo>::add_time_value(const Real t) const {
  const int varid = name_varid_map.at("time");
  size_t next_time_idx = 0;
  int retval = nc_inq_dimlen(ncid, time_dimid, &next_time_idx);
  CHECK_NCERR(retval);
  retval = nc_put_var1(ncid, varid, &next_time_idx, &t);
  CHECK_NCERR(retval);
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
  int retval = nc_def_dim(ncid, "coord", Geo::ndim, &coord_dimid);
  CHECK_NCERR(retval);
  n_nc_dims++;
}

template <typename Geo> template <FieldLocation FL>
void NcWriter<Geo>::define_scalar_field(const ScalarField<FL>& s) {
  LPM_ASSERT(ncid != NC_EBADID);
  LPM_ASSERT(time_dimid != NC_EBADID);
  int m_ndims = 2;
  int dimids[2];
  dimids[0] = time_dimid;
  std::string loc_string;
  switch (FL) {
    case( ParticleField ) : {
      LPM_REQUIRE(particles_dimid != NC_EBADID);
      dimids[1] = particles_dimid;
      break;
    }
    case( VertexField ) : {
      LPM_REQUIRE(vertices_dimid != NC_EBADID);
      dimids[1] = vertices_dimid;
      break;
    }
    case( EdgeField ) : {
      LPM_REQUIRE(edges_dimid != NC_EBADID);
      dimids[1] = edges_dimid;
      break;
    }
    case( FaceField ) : {
      LPM_REQUIRE(faces_dimid != NC_EBADID);
      dimids[1] = faces_dimid;
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
  retval = nc_put_att(ncid, NC_GLOBAL, "edges.n_max", nc_index_type::value, 1, &nmaxedges);
  CHECK_NCERR(retval);
  retval = nc_put_att(ncid, NC_GLOBAL, "edges.n_leaves", nc_index_type::value, 1, &nleaves);
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
      const auto kid_idx = edges.kid_host(i,j);
      retval = nc_put_var1(ncid, kids_varid, idx, &kid_idx);
      CHECK_NCERR(retval);
    }
  }

}

template <typename Geo> template <typename FaceType>
void NcWriter<Geo>::define_faces(const Faces<FaceType, Geo>& faces) {
  LPM_ASSERT(ncid != NC_EBADID);
  LPM_REQUIRE_MSG(faces_dimid == NC_EBADID, "faces dimension already defined.");
  LPM_REQUIRE(facekind_dimid == NC_EBADID);
  LPM_REQUIRE(coord_dimid != NC_EBADID);
  int retval = nc_def_dim(ncid, "faces", faces.nh(), &faces_dimid);
  CHECK_NCERR(retval);
  n_nc_dims++;

  retval = nc_def_dim(ncid, "four", 4, &four_dimid);
  CHECK_NCERR(retval);
  n_nc_dims++;

  retval = nc_def_dim(ncid, "facekind", FaceType::nverts, &facekind_dimid);
  CHECK_NCERR(retval);
  n_nc_dims++;

  const auto nmaxfaces = faces.n_max();
  const auto nfaces = faces.nh();
  const auto nleaves = faces.n_leaves_host();

  int mask_varid = NC_EBADID;
  int verts_varid = NC_EBADID;
  int edges_varid = NC_EBADID;
  int phys_crd_varid = NC_EBADID;
  int lag_crd_varid = NC_EBADID;
  int level_varid = NC_EBADID;
  int parents_varid = NC_EBADID;
  int kids_varid = NC_EBADID;
  int area_varid = NC_EBADID;
  int crd_inds_varid = NC_EBADID;

  retval = nc_def_var(ncid, "faces.mask", NC_UBYTE, 1, &faces_dimid, &mask_varid);
  CHECK_NCERR(retval);
  name_varid_map.emplace("faces.mask", mask_varid);

  const int vert_and_edge_dims[2] = {faces_dimid, facekind_dimid};
  retval = nc_def_var(ncid, "faces.vertices", nc_index_type::value, 2, vert_and_edge_dims, &verts_varid);
  CHECK_NCERR(retval);
  name_varid_map.emplace("faces.vertices", verts_varid);

  retval = nc_def_var(ncid, "faces.edges", nc_index_type::value, 2, vert_and_edge_dims,
    &edges_varid);
  CHECK_NCERR(retval);
  name_varid_map.emplace("faces.edges", edges_varid);

  retval = nc_def_var(ncid, "faces.crd_inds", nc_index_type::value, 1, &faces_dimid, &crd_inds_varid);
  CHECK_NCERR(retval);
  name_varid_map.emplace("faces.crd_inds", crd_inds_varid);

  retval = nc_def_var(ncid, "faces.level", NC_INT, 1, &faces_dimid, &level_varid);
  CHECK_NCERR(retval);
  name_varid_map.emplace("faces.level", level_varid);

  retval = nc_def_var(ncid, "faces.parent", nc_index_type::value, 1, &faces_dimid, &parents_varid);
  CHECK_NCERR(retval);
  name_varid_map.emplace("faces.parent", parents_varid);

  const int kid_dims[2] = {faces_dimid, four_dimid};
  retval = nc_def_var(ncid, "faces.kids", nc_index_type::value, 2, kid_dims, &kids_varid);
  CHECK_NCERR(retval);
  name_varid_map.emplace("faces.kids", kids_varid);
  retval = nc_put_att(ncid, NC_GLOBAL, "faces.n_leaves", nc_index_type::value, 1, &nleaves);
  CHECK_NCERR(retval);
  retval = nc_put_att(ncid, NC_GLOBAL, "faces.n_max", nc_index_type::value, 1, &nmaxfaces);
  CHECK_NCERR(retval);

  const int pcrd_dims[3] = {time_dimid, faces_dimid, coord_dimid};
  retval = nc_def_var(ncid, "faces.phys_crds", nc_real_type::value, 3, pcrd_dims, &phys_crd_varid);
  CHECK_NCERR(retval);
  name_varid_map.emplace("faces.phys_crds", phys_crd_varid);

  const int lcrd_dims[2] = {faces_dimid, coord_dimid};
  retval = nc_def_var(ncid, "faces.lag_crds", nc_real_type::value, 2, lcrd_dims,
    &lag_crd_varid);
  CHECK_NCERR(retval);
  name_varid_map.emplace("faces.lag_crds", lag_crd_varid);

  retval = nc_def_var(ncid, "faces.area", nc_real_type::value, 1, &faces_dimid, &area_varid);
  CHECK_NCERR(retval);
  name_varid_map.emplace("faces.area", area_varid);
  const auto area_unit_str = ekat::units::to_string(ekat::units::m * ekat::units::m);
  retval = nc_put_att_text(ncid, area_varid, "units", area_unit_str.size(), area_unit_str.c_str());


  for (size_t i=0; i<nfaces; ++i) {
    const size_t mask_idx = i;
    const uint_fast8_t mask_val = (faces._hmask(i) ? 1 : 0);
    retval = nc_put_var1(ncid, mask_varid, &mask_idx, &mask_val);
    CHECK_NCERR(retval);
    for (size_t j=0; j<4; ++j) {
      const size_t kids_idx[2] = {i,j};
      retval = nc_put_var1(ncid, kids_varid, kids_idx, &faces._hostkids(i,j));
      CHECK_NCERR(retval);
    }
    for (size_t j=0; j<FaceType::nverts; ++j) {
      const size_t vert_and_edge_idx[2] = {i,j};
      retval = nc_put_var1(ncid, verts_varid, vert_and_edge_idx, &faces._hostverts(i,j));
      CHECK_NCERR(retval);
      retval = nc_put_var1(ncid, edges_varid, vert_and_edge_idx, &faces._hostedges(i,j));
    }
    for (size_t j=0; j<Geo::ndim; ++j) {
      const size_t pcrd_idx[3] = {0, i, j};
      const size_t lcrd_idx[2] = {i,j};
      const Real pcrd_val = faces.phys_crds->get_crd_component_host(i,j);
      const Real lcrd_val = faces.lag_crds->get_crd_component_host(i,j);
      retval = nc_put_var1(ncid, phys_crd_varid, pcrd_idx, &pcrd_val);
      CHECK_NCERR(retval);
      retval = nc_put_var1(ncid, lag_crd_varid, lcrd_idx, &lcrd_val);
      CHECK_NCERR(retval);
    }
  }

  const size_t start = 0;
  const size_t count = nfaces;
  retval = nc_put_vara(ncid, crd_inds_varid, &start, &count, faces._host_crd_inds.data());
  CHECK_NCERR(retval);
  retval = nc_put_vara(ncid, parents_varid, &start, &count, faces._hostparent.data());
  CHECK_NCERR(retval);
  retval = nc_put_vara(ncid, area_varid, &start, &count, faces._hostarea.data());
  CHECK_NCERR(retval);
  retval = nc_put_vara(ncid, level_varid, &start, &count, faces._hlevel.data());
  CHECK_NCERR(retval);

}

template <typename Geo>
void NcWriter<Geo>::update_crds(const size_t time_idx, const int varid, const Coords<Geo>& crds) {
  LPM_ASSERT(ncid != NC_EBADID);
  LPM_REQUIRE(varid != NC_EBADID);
  LPM_REQUIRE_MSG(time_idx < n_timesteps(), "time variable must be defined before adding timestep data.");

  for (size_t i=0; i<crds.nh(); ++i) {
    for (size_t j=0; j<Geo::ndim; ++j) {
      const size_t idx[3] = {time_idx, i, j};
      const Real crd_val = crds.get_crd_component_host(i,j);
      int retval = nc_put_var1(ncid, varid, idx, &crd_val);
      CHECK_NCERR(retval);
    }
  }
}

template <typename Geo> template <typename SeedType>
void NcWriter<Geo>::define_polymesh(const PolyMesh2d<SeedType>& mesh) {
  define_vertices(mesh.vertices);
  define_edges(mesh.edges);
  define_faces(mesh.faces);

  int varid = NC_EBADID;
  int* ignore_me;
  int retval = nc_def_var(ncid, "base_tree_depth", NC_INT, 0, ignore_me, &varid);
  CHECK_NCERR(retval);
  name_varid_map.emplace("base_tree_depth", varid);
  const auto mesh_id_str = SeedType::id_string();
  retval = nc_put_att_text(ncid, NC_GLOBAL, "MeshSeed", mesh_id_str.size(), mesh_id_str.c_str());
  CHECK_NCERR(retval);
}

template <typename Geo>
void NcWriter<Geo>::update_particle_phys_crds(const size_t time_idx, const Coords<Geo>& pcrds) {
  //TODO after particle class is defined
}

template <typename Geo>
void NcWriter<Geo>::update_vertex_phys_crds(const size_t time_idx, const Vertices<Coords<Geo>>& verts) {
  update_crds(time_idx, name_varid_map.at("vertices.phys_crds"), *(verts.phys_crds));
}

template <typename Geo> template <FieldLocation FL>
void NcWriter<Geo>::put_scalar_field(const size_t time_idx, const ScalarField<FL>& s) {
  LPM_REQUIRE(time_idx < n_timesteps());
  const int varid = name_varid_map.at(s.name);
  const size_t start[2] = {time_idx, 0};
  size_t count[2];
  count[0] = 1;
  switch (FL) {
    case (ParticleField) : {
      count[1] = n_particles();
      break;
    }
    case( VertexField ) : {
      count[1] = n_vertices();
      break;
    }
    case( EdgeField ) : {
      count[1] = n_edges();
      break;
    }
    case( FaceField ) : {
      count[1] = n_faces();
      break;
    }
  }
  int retval = nc_put_vara(ncid, varid, start, count, s.hview.data());
  CHECK_NCERR(retval);
}

template <typename Geo> template <FieldLocation FL>
void NcWriter<Geo>::put_vector_field(const size_t time_idx, const VectorField<Geo,FL>& v) {
  LPM_REQUIRE(time_idx < n_timesteps());

  const int varid = name_varid_map.at(v.name);

  Index n_entries;
  switch (FL) {
    case (ParticleField) : {
      n_entries = n_particles();
      break;
    }
    case( VertexField ) : {
      n_entries = n_vertices();
      break;
    }
    case( EdgeField ) : {
      n_entries = n_edges();
      break;
    }
    case( FaceField ) : {
      n_entries = n_faces();
      break;
    }
  }
  for (size_t i=0; i<n_entries; ++i) {
    for (size_t j=0; j<Geo::ndim; ++j) {
      const size_t idx[3] = {time_idx, i, j};
      int retval = nc_put_var1(ncid, varid, idx, &v.hview(i,j));
      CHECK_NCERR(retval);
    }
  }
}

template <typename Geo> template <typename FaceType>
void NcWriter<Geo>::update_face_phys_crds(const size_t time_idx, const Faces<FaceType, Geo>& faces) {
  update_crds(time_idx, name_varid_map.at("faces.phys_crds"), *(faces.phys_crds));
  const size_t start[2] = {time_idx, 0};
  const size_t count[2] = {1, faces.nh()};
  int retval = nc_put_vara(ncid, name_varid_map.at("faces.area"), start, count, faces._hostarea.data());
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
    retval = nc_put_att(ncid, NC_GLOBAL, "vertices.n_max", nc_index_type::value, 1, &nmaxverts);
    CHECK_NCERR(retval);
  }
  {
    int varid = NC_EBADID;
    const int m_ndims = 3;
    const int dimids[3] = {time_dimid, vertices_dimid, coord_dimid};
    retval = nc_def_var(ncid, "vertices.phys_crds", nc_real_type::value, m_ndims, dimids, &varid);
    CHECK_NCERR(retval);
    name_varid_map.emplace("vertices.phys_crds", varid);
    for (size_t i=0; i<nverts; ++i) {
      for (size_t j=0; j<Geo::ndim; ++j) {
        const size_t idx[3] = {0, i, j};
        const Real crd_val = vertices.phys_crds->get_crd_component_host(h_inds(i),j);
        retval = nc_put_var1(ncid, varid, idx, &crd_val);
        CHECK_NCERR(retval);
      }
    }
  }
  {
    int varid = NC_EBADID;
    const int m_ndims = 2;
    const int dimids[2] = {vertices_dimid, coord_dimid};
    retval = nc_def_var(ncid, "vertices.lag_crds", nc_real_type::value, m_ndims, dimids, &varid);
    CHECK_NCERR(retval);
    name_varid_map.emplace("vertices.lag_crds", varid);
    for (size_t i=0; i<nverts; ++i) {
      for (size_t j=0; j<Geo::ndim; ++j) {
        const size_t idx[2] = {i, j};
        const Real crd_val = vertices.lag_crds->get_crd_component_host(h_inds(i),j);
        retval = nc_put_var1(ncid, varid, idx, &crd_val);
        CHECK_NCERR(retval);
      }
    }
  }
  {
    //TODO: If verts are dual...
  }
}


template <typename Geo>
std::string NcWriter<Geo>::info_string(const int tab_level) const {
  auto tabstr = indent_string(tab_level);
  std::ostringstream ss;
  ss << tabstr << "NcWriter info:\n";
  tabstr += "\t";
  ss << tabstr << "filename: " << fname << "\n";
  ss << tabstr << "ncid: " << ncid << "\n";
  ss << tabstr << "n_nc_dims: " << n_nc_dims << "\n";
  ss << tabstr << "variables:\n";
  for (auto& v : name_varid_map) {
    ss << "\t" << v.first << "\n";
  }
  return ss.str();
}

template <typename Geo> template <FieldLocation FL>
void NcWriter<Geo>::define_vector_field(const VectorField<Geo,FL>& v) {
  LPM_ASSERT(ncid != NC_EBADID);
  LPM_ASSERT(time_dimid != NC_EBADID);
  LPM_ASSERT(coord_dimid != NC_EBADID);
  int m_ndims = 3;
  int dimids[3];
  dimids[0] = time_dimid;
  dimids[2] = coord_dimid;
  switch (FL) {
    case( ParticleField ) : {
      LPM_REQUIRE(particles_dimid != NC_EBADID);
      dimids[1] = particles_dimid;
      break;
    }
    case( VertexField ) : {
      LPM_REQUIRE(vertices_dimid != NC_EBADID);
      dimids[1] = vertices_dimid;
      break;
    }
    case( EdgeField ) : {
      LPM_REQUIRE(edges_dimid != NC_EBADID);
      dimids[1] = edges_dimid;
      break;
    }
    case( FaceField ) : {
      LPM_REQUIRE(faces_dimid != NC_EBADID);
      dimids[1] = faces_dimid;
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

template <typename SeedType>
std::shared_ptr<PolyMesh2d<SeedType>> PolymeshReader::init_polymesh() {
  auto result = std::shared_ptr<PolyMesh2d<SeedType>>(new PolyMesh2d<SeedType>(nmaxverts, nmaxedges, nmaxfaces));
  fill_vertices(result->vertices);
  fill_edges(result->edges);
  fill_faces(result->faces);
  fill_crds(*(result->vertices.phys_crds), *(result->vertices.lag_crds),
    *(result->faces.phys_crds), *(result->faces.lag_crds));
  result->update_device();
  return result;
}

template <typename Geo>
void PolymeshReader::fill_vertices(Vertices<Coords<Geo>>& verts) {
  LPM_ASSERT(vertices_dimid != NC_EBADID);
  const Index nverts = n_vertices();
  verts._nh() = nverts;
  auto h_vert_crds = verts.host_crd_inds();
  const size_t start = 0;
  const size_t count = nverts;
  int retval = nc_get_vara(ncid, name_varid_map.at("vertices.crd_inds"), &start, &count, h_vert_crds.data());
  CHECK_NCERR(retval);
}

template <typename FaceType, typename Geo>
void PolymeshReader::fill_faces(Faces<FaceType, Geo>& faces) {
  LPM_ASSERT(faces_dimid != NC_EBADID);
  LPM_ASSERT(facekind_dimid != NC_EBADID);
  size_t nfaceverts;
  int retval = nc_inq_dimlen(ncid, facekind_dimid, &nfaceverts);
  CHECK_NCERR(retval);
  LPM_REQUIRE(nfaceverts == FaceType::nverts);

  const Index nfaces = n_faces();
  Index nleaves;
  retval = nc_get_att(ncid, NC_GLOBAL, "faces.n_leaves", &nleaves);
  faces._nh() = nfaces;
  faces._hn_leaves() = nleaves;
  {
    const size_t start = 0;
    const size_t count = nfaces;
    retval = nc_get_vara(ncid, name_varid_map.at("faces.crd_inds"), &start, &count, faces._host_crd_inds.data());
    CHECK_NCERR(retval);
    retval = nc_get_vara(ncid, name_varid_map.at("faces.parent"), &start, &count, faces._hostparent.data());
    CHECK_NCERR(retval);
    retval = nc_get_vara(ncid, name_varid_map.at("faces.level"), &start, &count, faces._hlevel.data());
    CHECK_NCERR(retval);
    retval = nc_get_vara(ncid, name_varid_map.at("faces.area"), &start, &count, faces._hostarea.data());
  }
  for (size_t i=0; i<nfaces; ++i) {
    size_t idx1 = i;
    uint_fast8_t mask_val;
    retval = nc_get_var1(ncid, name_varid_map.at("faces.mask"), &idx1, &mask_val);
    CHECK_NCERR(retval);
    faces._hmask(i) = (mask_val > 0 ? true : false);
    for (size_t j=0; j<nfaceverts; ++j) {
      size_t idx2[2] = {i,j};
      retval = nc_get_var1(ncid, name_varid_map.at("faces.vertices"), idx2, &faces._hostverts(i,j));
      CHECK_NCERR(retval);
      retval = nc_get_var1(ncid, name_varid_map.at("faces.edges"), idx2, &faces._hostedges(i,j));
      CHECK_NCERR(retval);
    }
    for (size_t j=0; j<4; ++j) {
      size_t idx3[2] = {i,j};
      retval = nc_get_var1(ncid, name_varid_map.at("faces.kids"), idx3, &faces._hostkids(i,j));
    }
  }
}

template <typename Geo>
void PolymeshReader::fill_crds(Coords<Geo>& vert_phys_crds, Coords<Geo>& vert_lag_crds,
  Coords<Geo>& face_phys_crds, Coords<Geo>& face_lag_crds) {

  const Index nverts = n_vertices();
  vert_phys_crds._nh() = nverts;
  vert_lag_crds._nh() = nverts;

  const size_t last_time_idx = n_timesteps()-1;

  for (size_t i=0; i<nverts; ++i) {
    for (size_t j=0; j<Geo::ndim; ++j) {
      const size_t pidx[3] = {last_time_idx, i,j};
      const size_t lidx[2] = {i,j};
      Real pcrdval;
      Real lcrdval;
      int retval = nc_get_var1(ncid, name_varid_map.at("vertices.phys_crds"), pidx, &pcrdval);
      CHECK_NCERR(retval);
      retval = nc_get_var1(ncid, name_varid_map.at("vertices.lag_crds"), lidx, &lcrdval);
      CHECK_NCERR(retval);
      vert_phys_crds._hostcrds(i,j) = pcrdval;
      vert_lag_crds._hostcrds(i,j) = lcrdval;
    }
  }

  const Index nfaces = n_faces();
  face_phys_crds._nh() = nfaces;
  face_lag_crds._nh() = nfaces;

  for (size_t i=0; i<nfaces; ++i) {
    for (size_t j=0; j<Geo::ndim; ++j) {
      const size_t pidx[3] = {last_time_idx, i,j};
      const size_t lidx[2] = {i,j};
      Real pcrdval;
      Real lcrdval;
      int retval = nc_get_var1(ncid, name_varid_map.at("faces.phys_crds"), pidx, &pcrdval);
      CHECK_NCERR(retval);
      retval = nc_get_var1(ncid, name_varid_map.at("faces.lag_crds"), lidx, &lcrdval);
      CHECK_NCERR(retval);
      face_phys_crds._hostcrds(i,j) = pcrdval;
      face_lag_crds._hostcrds(i,j) = lcrdval;
    }
  }

}

// ETI
template class NcWriter<PlaneGeometry>;
template class NcWriter<SphereGeometry>;

}
#endif

