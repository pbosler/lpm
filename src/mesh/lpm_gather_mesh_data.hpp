#ifndef LPM_GATHER_MESH_DATA_HPP
#define LPM_GATHER_MESH_DATA_HPP

#include "LpmConfig.h"
#include "lpm_coords.hpp"
#include "lpm_field.hpp"
#include "mesh/lpm_polymesh2d.hpp"

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
  PolyMesh2d<SeedType>& mesh;
  std::map<std::string, scalar_view_type> scalar_fields;
  std::map<std::string, typename SeedType::geo::vec_view_type> vector_fields;
  bool unpacked;
  scalar_view_type x;
  scalar_view_type y;
  scalar_view_type z;
  scalar_view_type lag_x;
  scalar_view_type lag_y;
  scalar_view_type lag_z;

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

  void unpack_coordinates();

  explicit GatherMeshData(PolyMesh2d<SeedType>& pm);

  explicit GatherMeshData(const PlanarGrid& grid);

  inline Index n() const { return x.extent(0); }

  std::string info_string(const Int tab_lev = 0,
                          const bool verbose = false) const;

  void update_host() const;

  void update_device() const;

  void gather_coordinates();

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

  void gather_vector_fields(
      const std::map<std::string, VectorField<typename SeedType::geo,
                                              VertexField>>& vert_fields,
      const std::map<std::string,
                     VectorField<typename SeedType::geo, FaceField>>&
          face_fields);

 private:
  bool _host_initialized;
  template <int ndim>
  typename std::enable_if<ndim==2, void>::type unpack_helper();
  template <int ndim>
  typename std::enable_if<ndim==3, void>::type unpack_helper();
};

}  // namespace Lpm

#endif
