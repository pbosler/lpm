#ifndef LPM_SCATTER_MESH_DATA
#define LPM_SCATTER_MESH_DATA

#include <map>

#include "LpmConfig.h"
#include "lpm_field.hpp"
#include "mesh/lpm_gather_mesh_data.hpp"

namespace Lpm {

template <typename ViewType>
struct ScatterFaceLeafData {
  ViewType mesh_face_values; /// output
  ViewType leaf_face_values; /// input
  mask_view_type face_mask; /// input
  index_view_type face_leaf_view; /// input

  ScatterFaceLeafData(ViewType fv, const ViewType lv,
    const mask_view_type fm, const index_view_type fl) :
    mesh_face_values(fv),
    leaf_face_values(lv),
    face_mask(fm),
    face_leaf_view(fl) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index i) const {
    if (!face_mask(i)) {
      for (int j=0; j<mesh_face_values.extent(1); ++j) {
        mesh_face_values(i,j) = leaf_face_values(face_leaf_view(i), j);
      }
    }
  }
};

template <>
struct ScatterFaceLeafData<scalar_view_type> {
  scalar_view_type mesh_face_values;
  scalar_view_type leaf_face_values;
  mask_view_type face_mask;
  index_view_type face_leaf_view;

  ScatterFaceLeafData(scalar_view_type fv, const scalar_view_type lv,
    const mask_view_type fm, const index_view_type fl) :
    mesh_face_values(fv),
    leaf_face_values(lv),
    face_mask(fm),
    face_leaf_view(fl) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index i) const {
    mesh_face_values(i) = (face_mask(i) ? 0 : leaf_face_values(face_leaf_view(i)));
  }
};


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


  std::string info_string(const int tab_lev = 0) const;
};

}  // namespace Lpm

#endif
