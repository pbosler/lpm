#ifndef LPM_SCATTER_MESH_DATA
#define LPM_SCATTER_MESH_DATA

#include <map>

#include "LpmConfig.h"
#include "lpm_field.hpp"
#include "mesh/lpm_gather_mesh_data.hpp"

namespace Lpm {

/// Scatters gathered output back to a PolyMesh2d.
template <typename SeedType>
struct ScatterMeshData {
  using vert_scalar_map = std::map<std::string, ScalarField<VertexField>>;
  using vert_vector_map =
      std::map<std::string, VectorField<typename SeedType::geo, VertexField>>;
  using face_scalar_map = std::map<std::string, ScalarField<FaceField>>;
  using face_vector_map =
      std::map<std::string, VectorField<typename SeedType::geo, FaceField>>;

  /// gathered data that needs to be put back onto a mesh
  const GatherMeshData<SeedType>& output;
  /// the output mesh
  PolyMesh2d<SeedType>& mesh;

  /// constructor.
  /// @param out gathered data, ready to be moved to a mesh
  /// @param pm output mesh
  ScatterMeshData(const GatherMeshData<SeedType>& out,
                  PolyMesh2d<SeedType>& pm);

  /// scatters Lagrangian coordinate data from the gathered source to the mesh
  /// target
  void scatter_lag_crds();

  /// scatters physical coordinate data from the gathered source to the mesh target
  void scatter_phys_crds();

  /// scatters fields from gathered source to mesh target
  void scatter_fields(
      const vert_scalar_map& vertex_scalar_fields,
      const face_scalar_map& face_scalar_fields,
      const vert_vector_map& vertex_vector_fields = vert_vector_map(),
      const face_vector_map& face_vector_fields = face_vector_map());
};

}  // namespace Lpm

#endif
