#ifndef LPM_NETCDF_HPP
#define LPM_NETCDF_HPP

#include "LpmConfig.h"

#ifdef LPM_USE_NETCDF

#include "lpm_assert.hpp"
#include <netcdf.h>
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

// index type in netCDF namespace
typedef std::conditional<std::is_same<int,Index>::value,
  NcInt, NcInt64>::type nc_index_type;

// real type in netCDF namespace
typedef std::conditional<std::is_same<double,Real>::value,
  NcDouble, NcFloat>::type nc_real_type;

// returns true if filename ends in ".nc"
bool has_nc_file_extension(const std::string& filename);

// decodes integer return values to netcdf error codes
std::string nc_decode_error(const int& ec, const std::string& file, const std::string& fn, const int& line);

#define CHECK_NCERR(ec) nc_decode_error(ec, __FILE__, __FUNCTION__, __LINE__)

/// key-value pairs for metadata attributes
typedef std::pair<std::string, std::string> text_att_type;

template <int n_horiz_dims>
struct NcDataLayoutTraits {
  typedef std::array<Index, n_horiz_dims> horizontal_index_type;
  typedef std::array<Index, n_horiz_dims+1> spatial_index_type;
};

// template <typename Geo>
// class NcWriter {
//   public:
//
//     explicit NcWriter(const std::string& filename) :
//       fname(filename),
//       ncid(NC_EBADID),
//       time_dimid(NC_EBADID),
//       particles_dimid(NC_EBADID),
//       vertices_dimid(NC_EBADID),
//       edges_dimid(NC_EBADID),
//       faces_dimid(NC_EBADID),
//       coord_dimid(NC_EBADID),
//       two_dimid(NC_EBADID),
//       four_dimid(NC_EBADID),
//       facekind_dimid(NC_EBADID),
//       n_nc_dims(0),
//       name_varid_map() {
//       LPM_REQUIRE_MSG(has_nc_file_extension(filename), "NcWriter error: filename '"
//         + filename + "' has invalid extension (must be .nc)");
//       open();
//       define_time_dim();
//       define_coord_dim();
//     }
//
//     ~NcWriter() {close();}
//
//     void define_file_attribute(const text_att_type& att_pair) const;
//
//     std::string info_string(const int tab_level = 0) const;
//
//     void define_particles_dim(const Index np);
//
//     void define_vertices(const Vertices<Coords<Geo>>& verts);
//
//     void define_edges(const Edges& edges);
//
//     template <typename FaceType>
//     void define_faces(const Faces<FaceType,Geo>& faces);
//
//     void define_single_real_var(const std::string& name, const ekat::units::Units& units,
//       const Real val, const std::vector<text_att_type> metadata=std::vector<text_att_type>());
//
//     Index n_particles() const;
//
//     Index n_timesteps() const;
//
//     Index n_vertices() const;
//
//     Index n_edges() const;
//
//     Index n_faces() const;
//
//     void add_time_value(const Real t) const;
//
//     void update_particle_phys_crds(const size_t time_idx, const Coords<Geo>& pcrds);
//
//     void update_vertex_phys_crds(const size_t time_idx, const Vertices<Coords<Geo>>& verts);
//
//     template <typename FaceType>
//     void update_face_phys_crds(const size_t time_idx, const Faces<FaceType,Geo>& faces);
//
//     template <typename SeedType>
//     void define_polymesh(const PolyMesh2d<SeedType>& mesh);
//
//     template <FieldLocation FL>
//     void define_scalar_field(const ScalarField<FL>& s);
//
//     template <FieldLocation FL>
//     void define_vector_field(const VectorField<Geo, FL>& v);
//
//     template <FieldLocation FL>
//     void put_scalar_field(const size_t time_idx, const ScalarField<FL>& s);
//
//     template <FieldLocation FL>
//     void put_vector_field(const size_t time_idx, const VectorField<Geo, FL>& v);
//
//   protected:
//     void handle_errcode(const int& ec, const std::string& file="",
//       const std::string& fn="", const int& line=constants::NULL_IND) const;
//
//     void open();
//
//     void close();
//
//     void update_crds(const size_t time_idx, const int varid, const Coords<Geo>& crds);
//
//     void define_time_dim();
//
//     void define_coord_dim();
//
//     std::string fname;
//     int ncid;
//     int time_dimid;
//     int particles_dimid;
//     int vertices_dimid;
//     int edges_dimid;
//     int faces_dimid;
//     int facekind_dimid;
//     int coord_dimid;
//     int n_nc_dims;
//     int two_dimid;
//     int four_dimid;
//
//     std::map<std::string, int> name_varid_map;
// };



// class PolymeshReader : public NcReader {
//   public:
//     explicit PolymeshReader(const std::string& full_filename) :
//       NcReader(full_filename),
//       vertices_dimid(NC_EBADID),
//       edges_dimid(NC_EBADID),
//       faces_dimid(NC_EBADID),
//       facekind_dimid(NC_EBADID)
//       {
//         init_dims();
//       }
//
//     virtual ~PolymeshReader() {}
//
//     Index n_vertices() const;
//
//     Index n_edges() const;
//
//     Index n_faces() const;
//
//     template <typename SeedType>
//     std::shared_ptr<PolyMesh2d<SeedType>> init_polymesh();
//
//     std::string info_string(const int tab_level=0) const override;
//
//   protected:
//     Index nmaxverts;
//     Index nmaxedges;
//     Index nmaxfaces;
//     Int base_tree_depth;
//
//     void init_dims();
//
//     template <typename Geo>
//     void fill_vertices(Vertices<Coords<Geo>>& verts);
//
//     void fill_edges(Edges& edges);
//
//     template <typename FaceType, typename Geo>
//     void fill_faces(Faces<FaceType, Geo>& faces);
//
//     template <typename Geo>
//     void fill_crds(Coords<Geo>& vert_phys_crds, Coords<Geo>& vert_lag_crds,
//         Coords<Geo>& face_phys_crds, Coords<Geo>& face_lag_crds);
//
//
//     int vertices_dimid;
//     int edges_dimid;
//     int faces_dimid;
//     int facekind_dimid;
// };



}// namespace Lpm
#endif
#endif
