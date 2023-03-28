#ifndef LPM_SCATTER_MESH_DATA
#define LPM_SCATTER_MESH_DATA

#include <map>

#include "LpmConfig.h"
#include "lpm_field.hpp"
#include "mesh/lpm_gather_mesh_data.hpp"

namespace Lpm {

template <typename SeedType>
struct ScatterMeshData {
  const GatherMeshData<SeedType>& output;
  std::shared_ptr<PolyMesh2d<SeedType>> mesh;

  ScatterMeshData(const GatherMeshData<SeedType>& out,
                  const std::shared_ptr<PolyMesh2d<SeedType>> pm);

  void scatter_lag_crds();

  void scatter_fields(
      const std::map<std::string, ScalarField<VertexField>>
          vertex_scalar_fields,
      const std::map<std::string, ScalarField<FaceField>> face_scalar_fields,
      const std::map<std::string,
                     VectorField<typename SeedType::geo, VertexField>>&
          vertex_vector_field = std::map<
              std::string, VectorField<typename SeedType::geo, VertexField>>(),
      const std::map<std::string,
                     VectorField<typename SeedType::geo, FaceField>>&
          face_vector_field = std::map<
              std::string, VectorField<typename SeedType::geo, FaceField>>());
};

}  // namespace Lpm

#endif
