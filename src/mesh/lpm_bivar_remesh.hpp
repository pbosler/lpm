#ifndef LPM_BIVAR_REMESH_HPP
#define LPM_BIVAR_REMESH_HPP

#include "LpmConfig.h"
#include "lpm_coriolis.hpp"
#include "lpm_geometry.hpp"
#include "fortran/lpm_bivar_interface.hpp"
#include "fortran/lpm_bivar_interface_impl.hpp"
#include "mesh/lpm_gather_mesh_data.hpp"
#include "mesh/lpm_gather_mesh_data_impl.hpp"
#include "mesh/lpm_refinement_flags.hpp"
#include "mesh/lpm_scatter_mesh_data.hpp"
#include "mesh/lpm_scatter_mesh_data_impl.hpp"

#include <map>

namespace Lpm {

template <typename SeedType> struct BivarRemesh {
  static_assert(std::is_same<typename SeedType::geo, PlaneGeometry>::value,
    "bivar only works for planar meshes with free boundaries");
  using in_out_map = std::map<std::string, std::string>;
  using scalar_view_map = std::map<std::string, scalar_view_type>;
  using vector_view_map = std::map<std::string, typename PlaneGeometry::vec_view_type>;

  PolyMesh2d<SeedType>& new_mesh;
  const PolyMesh2d<SeedType>& old_mesh;

  /** Constructor / initializer.

    @param [in/out] new_mesh PolyMesh2d upon whose vertices and faces new data will be defined (the destination)
    @param [in] old_mesh PolyMesh2d upon whose vertices and faces the data are currently defined (the source)
    @param [in] scalar_names list of scalar field names to be interpolated from source to destination
    @param [in] vector_names list of vector field names to be interpoalted from source to destination
  */
  BivarRemesh(PolyMesh2d<SeedType>& new_mesh, const PolyMesh2d<SeedType>& old_mesh,
    const std::vector<std::string>& scalar_names, const std::vector<std::string>& vector_names);

  /** Initialize and gather scalar field data values from old mesh.
  */
  void init_scalar_fields(
    const std::map<std::string, ScalarField<VertexField>>& old_vert_fields,
    const std::map<std::string, ScalarField<FaceField>>& old_face_fields,
    const std::map<std::string, ScalarField<VertexField>>& new_vert_fields,
    const std::map<std::string, ScalarField<FaceField>>& new_face_fields);

  /** Initialize and gather vector field data values from old mesh.
  */
  void init_vector_fields(
    const std::map<std::string,
      VectorField<PlaneGeometry, VertexField>>& old_vert_fields,
    const std::map<std::string,
      VectorField<PlaneGeometry, FaceField>>& old_face_fields,
   const std::map<std::string,
      VectorField<PlaneGeometry, VertexField>>& new_vert_fields,
    const std::map<std::string,
      VectorField<PlaneGeometry, FaceField>>& new_face_fields);

  /** Directly interpolate all data from old mesh to new mesh without
    any spatial refinement.
  */
  void uniform_direct_remesh(
    const std::map<std::string, ScalarField<VertexField>>& new_vert_scalars,
    const std::map<std::string, ScalarField<FaceField>>& new_face_scalars,
    const std::map<std::string,
      VectorField<PlaneGeometry, VertexField>>& new_vert_vectors,
    const std::map<std::string,
      VectorField<PlaneGeometry, FaceField>>& new_face_vectors);

  /** Interpolate data from old mesh to new mesh, without any
    spatial refinement.
    Use indirect interpolation to define Lagrangian data, then
    use direct interpolation for all other data.

    @param [in] functor_map <key, value> pairs where keys are std::strings
      naming Lagrangian scalar fields, and the values are the functors
      that define their initial data (e.g., from lpm_tracer_gallery.hpp
      or lpm_vorticity_gallery.hpp).
  */
  template <typename FunctorMapType>
  void uniform_indirect_remesh(FunctorMapType& functor_map,
    const std::map<std::string, ScalarField<VertexField>>& new_vert_scalars,
    const std::map<std::string, ScalarField<FaceField>>& new_face_scalars,
    const std::map<std::string,
      VectorField<PlaneGeometry, VertexField>>& new_vert_vectors,
    const std::map<std::string,
      VectorField<PlaneGeometry, FaceField>>& new_face_vectors);


  template <typename FunctorMapType, typename FlagType>
  void adaptive_indirect_remesh(FunctorMapType& functor_map, FlagType& flag,
    const std::map<std::string, ScalarField<VertexField>>& new_vert_scalars,
    const std::map<std::string, ScalarField<FaceField>>& new_face_scalars,
    const std::map<std::string,
      VectorField<PlaneGeometry, VertexField>>& new_vert_vectors,
    const std::map<std::string,
      VectorField<PlaneGeometry, FaceField>>& new_face_vectors);

  template <typename FunctorMapType, typename FlagType1, typename FlagType2>
  void adaptive_indirect_remesh(FunctorMapType& functor_map,
    FlagType1& flag1, FlagType2& flag2,
    const std::map<std::string, ScalarField<VertexField>>& new_vert_scalars,
    const std::map<std::string, ScalarField<FaceField>>& new_face_scalars,
    const std::map<std::string,
      VectorField<PlaneGeometry, VertexField>>& new_vert_vectors,
    const std::map<std::string,
      VectorField<PlaneGeometry, FaceField>>& new_face_vectors);

  template <typename FunctorMapType, typename FlagType1, typename FlagType2, typename FlagType3>
  void adaptive_indirect_remesh(FunctorMapType& functor_map,
    FlagType1& flag1, FlagType2& flag2, FlagType3& flag3,
    const std::map<std::string, ScalarField<VertexField>>& new_vert_scalars,
    const std::map<std::string, ScalarField<FaceField>>& new_face_scalars,
    const std::map<std::string,
      VectorField<PlaneGeometry, VertexField>>& new_vert_vectors,
    const std::map<std::string,
      VectorField<PlaneGeometry, FaceField>>& new_face_vectors);

  protected:
    std::unique_ptr<BivarInterface<SeedType>> bivar;
    std::unique_ptr<GatherMeshData<SeedType>> old_gather;
    std::unique_ptr<GatherMeshData<SeedType>> new_gather;
    std::unique_ptr<ScatterMeshData<SeedType>> new_scatter;

};


} // namespace Lpm

#endif
