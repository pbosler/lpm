#ifndef LPM_GATHER_MESH_DATA_HPP
#define LPM_GATHER_MESH_DATA_HPP

#include "LpmConfig.h"
#include "mesh/lpm_polymesh2d.hpp"
#include "lpm_field.hpp"

namespace Lpm {

struct PlanarGrid; // fwd decl

/**  Gather vertices and leaf face data (coordinates, scalar fields, and
  vector fields) into compressed arrays containing only the particles
  that are dynamically active (no faces that have been divided).

  This struct is the main entry point for clients.
*/
template <typename SeedType>
struct GatherMeshData {
  scalar_view_type x;
  scalar_view_type y;
  scalar_view_type z;
  std::shared_ptr<PolyMesh2d<SeedType>> mesh;
  std::map<std::string, scalar_view_type> scalar_fields;
  std::map<std::string, typename SeedType::geo::vec_view_type> vector_fields;

  typename scalar_view_type::HostMirror h_x;
  typename scalar_view_type::HostMirror h_y;
  typename scalar_view_type::HostMirror h_z;
  std::map<std::string, typename scalar_view_type::HostMirror> h_scalar_fields;
  std::map<std::string, typename SeedType::geo::vec_view_type::HostMirror> h_vector_fields;

  explicit GatherMeshData(const std::shared_ptr<PolyMesh2d<SeedType>> pm);

  explicit GatherMeshData(const PlanarGrid& grid);

  inline Index n() const {return x.extent(0);}

  std::string info_string(const Int tab_lev=0, const bool verbose=false) const;

  void update_host() const;

  void update_device() const;

  template <int ndim>
  typename std::enable_if<ndim==3, void>::type gather_coordinates();

  template <int ndim>
  typename std::enable_if<ndim==2, void>::type gather_coordinates();

  void init_scalar_field(const std::string& name, const scalar_view_type& view);

  void init_scalar_fields(
    const std::map<std::string, ScalarField<VertexField>>& vert_fields,
    const std::map<std::string, ScalarField<FaceField>>& face_fields);

  void init_vector_fields(
    const std::map<std::string,
      VectorField<typename SeedType::geo, VertexField>>& vert_fields,
    const std::map<std::string,
      VectorField<typename SeedType::geo, FaceField>>& face_fields);

  void gather_scalar_fields(
    const std::map<std::string, ScalarField<VertexField>>& vert_fields,
    const std::map<std::string, ScalarField<FaceField>>& face_fields);

  void gather_vector_fields(
    const std::map<std::string,
      VectorField<typename SeedType::geo, VertexField>>& vert_fields,
    const std::map<std::string,
      VectorField<typename SeedType::geo, FaceField>>& face_fields);

  private:
    bool _host_initialized;
};

} // namespace Lpm

#endif
