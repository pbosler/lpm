#ifndef LPM_NETCDF_HPP
#define LPM_NETCDF_HPP

#include "LpmConfig.h"

#include "mesh/lpm_polymesh2d.hpp"
#include "lpm_coords.hpp"
#include "lpm_field.hpp"
#include "mesh/lpm_vertices.hpp"
#include "mesh/lpm_edges.hpp"
#include "mesh/lpm_faces.hpp"
#include "lpm_assert.hpp"
#include "netcdf.h"
#include <memory>
#include <map>

namespace Lpm {

struct NcFloat {
  static constexpr int value = NC_FLOAT;
};

struct NcDouble {
  static constexpr int value = NC_DOUBLE;
};

struct NcInt {
  static constexpr int value = NC_INT;
};

struct NcInt64 {
  static constexpr int value = NC_INT64;
};

// returns true if filename ends in ".nc"
bool has_nc_file_extension(const std::string& filename);

// decodes integer return values to error codes
std::string nc_handle_errcode(const int& ec);

// index type in netCDF namespace
typedef std::conditional<std::is_same<int,Index>::value,
  NcInt, NcInt64>::type nc_index_type;

// real type in netCDF namespace
typedef std::conditional<std::is_same<double,Real>::value,
  NcDouble, NcFloat>::type nc_real_type;

#define CHECK_NCERR(ec) \
  if (ec != NC_NOERR) handle_errcode(ec, __FILE__, __FUNCTION__, __LINE__)

/// key-value pairs for metadata attributes
typedef std::pair<std::string, std::string> text_att_type;

template <typename Geo>
class NcWriter {
  public:

    NcWriter(const std::string& filename) :
      fname(filename),
      ncid(NC_EBADID),
      time_dimid(NC_EBADID),
      particles_dimid(NC_EBADID),
      vertices_dimid(NC_EBADID),
      edges_dimid(NC_EBADID),
      faces_dimid(NC_EBADID),
      coord_dimid(NC_EBADID),
      two_dimid(NC_EBADID),
      four_dimid(NC_EBADID),
      n_nc_dims(0),
      name_varid_map() {
      LPM_REQUIRE_MSG(has_nc_file_extension(filename), "NcWriter error: filename '"
        + filename + "' has invalid extension (must be .nc)");
      open();
      define_time_dim();
      define_coord_dim();
    }

    ~NcWriter() {close();}

    void define_file_attribute(const text_att_type& att_pair) const;

    std::string info_string(const int& tab_level = 0) const;

    void define_particles_dim(const Index np);

    void define_vertices(const Vertices<Coords<Geo>>& verts);

    void define_edges(const Edges& edges);

    template <typename FaceType>
    void define_faces(const Faces<FaceType,Geo>& faces);

    void define_single_real_var(const std::string& name, const ekat::units::Units& units,
      const Real val, const std::vector<text_att_type> metadata=std::vector<text_att_type>());

    Index n_particles() const;

    Index n_timesteps() const;

    Index n_vertices() const;

    Index n_edges() const;

    Index n_faces() const;

    void put_vertices(const Vertices<Coords<Geo>>& vertices);

    void put_edges(const Edges& edges);

    template <typename FaceKind>
    void put_faces(const Faces<FaceKind, Geo>& faces);

    template <typename SeedType>
    void put_polymesh(const std::shared_ptr<PolyMesh2d<SeedType>>& mesh);

    template <FieldLocation FL>
    void define_scalar_field(const ScalarField<FL>& s);

    template <FieldLocation FL>
    void define_vector_field(const VectorField<Geo, FL>& v);

    template <FieldLocation FL>
    void put_scalar_field(const std::string& field_name, const Int time_idx,
      const ScalarField<FL>& s);

    template <FieldLocation FL>
    void put_vector_field(const std::string& field_name, const Int time_idx,
      const VectorField<Geo, FL>& v);

  protected:
    void handle_errcode(const int& ec, const std::string& file="",
      const std::string& fn="", const int& line=constants::NULL_IND) const;

    void open();

    void close();

    void define_time_dim();

    void define_coord_dim();

    std::string fname;
    int ncid;
    int time_dimid;
    int particles_dimid;
    int vertices_dimid;
    int edges_dimid;
    int faces_dimid;
    int coord_dimid;
    int n_nc_dims;
    int two_dimid;
    int four_dimid;

    std::map<std::string, int> name_varid_map;
};

// class NcReader {
//   public:
//     typedef typename ko::View<Index*>::HostMirror host_index_view;
//     typedef typename scalar_view_type::HostMirror host_scalar_view;
//     typedef typename ko::View<Real**>::HostMirror host_vector_view;
//
//     NcReader(const std::string& filename);
//
//   protected:
//     std::string fname;
//     std::unique_ptr<const netCDF::NcFile> ncfile;
//
//     std::multimap<std::string, netCDF::NcDim> dims;
//     std::multimap<std::string, netCDF::NcVar> vars;
//     std::multimap<std::string, netCDF::NcGroupAtt> atts;
//
//     ko::View<Real**> getCrdView(const netCDF::NcVar& crd_var) const;
//     void fill_host_index_view(host_index_view& hv, const netCDF::NcVar& ind_var) const;
//     void fill_host_scalar_view(host_scalar_view& hv, const netCDF::NcVar& fvar) const;
//     void fill_host_vector_view(host_vector_view& hv, const netCDF::NcVar& fvar) const;
// };
//
// class PolyMeshReader : NcReader {
//   public:
//     typedef typename ko::View<Index*>::HostMirror host_index_view;
//     typedef typename mask_view_type::HostMirror host_mask_view;
//     typedef typename ko::View<Index*[3]>::HostMirror host_topo_view_tri;
//     typedef typename ko::View<Index*[4]>::HostMirror host_topo_view_quad;
//
//     PolyMeshReader(const std::string& filename) : NcReader(filename) {}
//
//     Int getTreeDepth() const;
//
//     ko::View<Real**> getVertPhysCrdView() const;
//     ko::View<Real**> getVertLagCrdView() const;
//
//     Index nEdges() const;
//     void fill_origs(host_index_view& hv) const;
//     void fill_dests(host_index_view& hv) const;
//     void fill_lefts(host_index_view& hv) const;
//     void fill_rights(host_index_view& hv) const;
//     void fill_edge_tree(host_index_view& hv,
//       typename ko::View<Index*[2]>::HostMirror& hk, Index& nleaves) const;
//
//
//     Index nFaces() const;
//     void fill_facemask(host_mask_view& hv) const;
//     void fill_face_connectivity(host_topo_view_tri& faceverts,
//       host_topo_view_tri& faceedges) const;
//     void fill_face_connectivity(host_topo_view_quad& faceverts,
//       host_topo_view_quad& faceedges) const;
//     void fill_face_centers(host_index_view& hv) const;
//     void fill_face_levels(host_index_view& hv) const;
//     void fill_face_tree(host_index_view& hp,
//       typename ko::View<Index*[4]>::HostMirror& hk, Index& nleaves) const;
//     void fill_face_area(typename scalar_view_type::HostMirror& hv) const;
//
//     ko::View<Real**> getFacePhysCrdView() const;
//     ko::View<Real**> getFaceLagCrdView() const;
//
//
//
// };

}// namespace Lpm
#endif
