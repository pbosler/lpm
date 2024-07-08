#ifndef LPM_GATHER_MESH_DATA_HPP
#define LPM_GATHER_MESH_DATA_HPP

#include "LpmConfig.h"
#include "lpm_coords.hpp"
#include "lpm_field.hpp"
#include "mesh/lpm_polymesh2d.hpp"

#include <map>

namespace Lpm {

struct PlanarGrid;  // fwd decl

/**  Gather vertices and leaf face data (coordinates, scalar fields, and
  vector fields) into compressed arrays containing only the particles
  that are dynamically active (no faces that have been divided).

  This struct is the main entry point for clients.
*/
template <typename SeedType>
class GatherMeshData final {
 public:
  typename SeedType::geo::crd_view_type phys_crds;
  typename SeedType::geo::crd_view_type lag_crds;
  /// input mesh with data to be gathered
  const PolyMesh2d<SeedType>& mesh;
  /// gathered scalar data
  std::map<std::string, scalar_view_type> scalar_fields;
  /// gathered vector data
  std::map<std::string, typename SeedType::geo::vec_view_type> vector_fields;
  /// boolean indicates whether coordinates are one rank 2 array (packed) or two
  /// rank 1 arrays (unpacked)
  bool unpacked;

  scalar_view_type
      x;  /// if unpacked, view of gathered x-coordinates; else: null
  scalar_view_type
      y;  /// if unpacked, view of gathered y-coordinates; else: null
  scalar_view_type
      z;  /// if unpacked, view of gathered z-coordinates; else: null
  scalar_view_type
      lag_x;  /// if unpacked, view of gathered x-coordinates; else: null
  scalar_view_type
      lag_y;  /// if unpacked, view of gathered y-coordinates; else: null
  scalar_view_type
      lag_z;  /// if unpacked, view of gathered z-coordinates; else: null

  typename scalar_view_type::HostMirror h_x;
  typename scalar_view_type::HostMirror h_y;
  typename scalar_view_type::HostMirror h_z;
  typename scalar_view_type::HostMirror h_lag_x;
  typename scalar_view_type::HostMirror h_lag_y;
  typename scalar_view_type::HostMirror h_lag_z;
  typename SeedType::geo::crd_view_type::HostMirror h_phys_crds;
  typename SeedType::geo::crd_view_type::HostMirror h_lag_crds;
  std::map<std::string, typename scalar_view_type::HostMirror> h_scalar_fields;
  std::map<std::string, typename SeedType::geo::vec_view_type::HostMirror>
      h_vector_fields;

  /// allocates rank-1 coordinate arrays and deep-copies coordinates
  void unpack_coordinates();

  /// constructor
  /// @param [in]
  explicit GatherMeshData(const PolyMesh2d<SeedType>& pm);

  inline Index n() const { return phys_crds.extent(0); }

  std::string info_string(const Int tab_lev = 0,
                          const bool verbose = false) const;

  void update_host() const;

  void update_device() const;

  void init_scalar_field(const std::string& name, const scalar_view_type& view);

  void init_scalar_fields(
      const std::map<std::string, ScalarField<VertexField>>& vert_fields,
      const std::map<std::string, ScalarField<FaceField>>& face_fields);

  void init_vector_fields(
      const std::map<std::string, VectorField<typename SeedType::geo,
                                              VertexField>>& vert_fields,
      const std::map<std::string,
                     VectorField<typename SeedType::geo, FaceField>>&
          face_fields);

  void gather_scalar_fields(
      const std::map<std::string, ScalarField<VertexField>>& vert_fields,
      const std::map<std::string, ScalarField<FaceField>>& face_fields);

  void gather_scalar_fields(
      const std::map<std::string, scalar_view_type>& passive_views,
      const std::map<std::string, scalar_view_type>& active_views);

  void gather_vector_fields(
      const std::map<std::string, VectorField<typename SeedType::geo,
                                              VertexField>>& vert_fields,
      const std::map<std::string,
                     VectorField<typename SeedType::geo, FaceField>>&
          face_fields);

  void gather_coordinates(typename SeedType::geo::crd_view_type passive_x,
                          typename SeedType::geo::crd_view_type active_x);

 private:
  void gather_coordinates();
  bool _host_initialized;
  template <int ndim>
  typename std::enable_if<ndim == 2, void>::type unpack_helper();
  template <int ndim>
  typename std::enable_if<ndim == 3, void>::type unpack_helper();
};

}  // namespace Lpm

#endif
