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
  using vert_scalar_field_map = std::map<std::string, ScalarField<VertexField>>;
  using vert_vector_field_map = std::map<std::string, VectorField<PlaneGeometry,VertexField>>;
  using face_scalar_field_map = std::map<std::string, ScalarField<FaceField>>;
  using face_vector_field_map = std::map<std::string, VectorField<PlaneGeometry,FaceField>>;
  using in_out_map = std::map<std::string, std::string>;

  PolyMesh2d<SeedType>& new_mesh;
  vert_scalar_field_map new_vert_scalars;
  face_scalar_field_map new_face_scalars;
  vert_vector_field_map new_vert_vectors;
  face_vector_field_map new_face_vectors;

  const PolyMesh2d<SeedType>& old_mesh;
  vert_scalar_field_map old_vert_scalars;
  face_scalar_field_map old_face_scalars;
  vert_vector_field_map old_vert_vectors;
  face_vector_field_map old_face_vectors;

  /** Constructor / initializer.

    @param [in/out] new_mesh PolyMesh2d upon whose vertices and faces new data will be defined (the destination) new_mesh will only be changed if AMR is used.
    @param [in] old_mesh PolyMesh2d upon whose vertices and faces the data are currently defined (the source)
    @param [in] old_vert_scalars <name, Field> map for vertex data defined on old mesh
    @param [in] old_face_scalars <name, Field> map for face data defined on old mesh
    @param [in] old_vert_vectors <name, Field> map for vertex data defined on old mesh
    @param [in] old_face_vectors <name, Field> map for face data defined on old mesh
    @param [in/out] new_vert_scalars <name, Field> map for vertex data defined on new mesh
    @param [in/out] new_face_scalars <name, Field> map for face data defined on new mesh
    @param [in/out] new_vert_vectors <name, Field> map for vertex data defined on new mesh
    @param [in/out] new_face_vectors <name, Field> map for face data defined on new mesh
  */
  BivarRemesh(PolyMesh2d<SeedType>& new_mesh,
     vert_scalar_field_map& new_vert_scalars,
     face_scalar_field_map& new_face_scalars,
     vert_vector_field_map& new_vert_vectors,
     face_vector_field_map& new_face_vectors,
     const PolyMesh2d<SeedType>& old_mesh,
     const vert_scalar_field_map& old_vert_scalars,
     const face_scalar_field_map& old_face_scalars,
     const vert_vector_field_map& old_vert_vectors,
     const face_vector_field_map& old_face_vectors);

  /** Directly interpolate all data from old mesh to new mesh without
    any spatial refinement.
  */
  void uniform_direct_remesh();

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
  void uniform_indirect_remesh(FunctorMapType& functor_map);

//
//   template <typename FunctorMapType, typename FlagType>
//   void adaptive_indirect_remesh(FunctorMapType& functor_map, FlagType& flag,
//     const vert_scalar_field_map& new_vert_scalars,
//     const face_scalar_field_map& new_face_scalars,
//     const std::map<std::string,
//       VectorField<PlaneGeometry, VertexField>>& new_vert_vectors,
//     const std::map<std::string,
//       VectorField<PlaneGeometry, FaceField>>& new_face_vectors);
//
//   template <typename FunctorMapType, typename FlagType1, typename FlagType2>
//   void adaptive_indirect_remesh(FunctorMapType& functor_map,
//     FlagType1& flag1, FlagType2& flag2,
//     const vert_scalar_field_map& new_vert_scalars,
//     const face_scalar_field_map& new_face_scalars,
//     const std::map<std::string,
//       VectorField<PlaneGeometry, VertexField>>& new_vert_vectors,
//     const std::map<std::string,
//       VectorField<PlaneGeometry, FaceField>>& new_face_vectors);
//
//   template <typename FunctorMapType, typename FlagType1, typename FlagType2, typename FlagType3>
//   void adaptive_indirect_remesh(FunctorMapType& functor_map,
//     FlagType1& flag1, FlagType2& flag2, FlagType3& flag3,
//     const vert_scalar_field_map& new_vert_scalars,
//     const face_scalar_field_map& new_face_scalars,
//     const std::map<std::string,
//       VectorField<PlaneGeometry, VertexField>>& new_vert_vectors,
//     const std::map<std::string,
//       VectorField<PlaneGeometry, FaceField>>& new_face_vectors);

  protected:
    std::unique_ptr<BivarInterface<SeedType>> bivar;
    std::unique_ptr<GatherMeshData<SeedType>> old_gather;
    std::unique_ptr<GatherMeshData<SeedType>> new_gather;
    std::unique_ptr<ScatterMeshData<SeedType>> new_scatter;

    in_out_map scalar_in_out_map;
    in_out_map vector_in_out_map;

    void build_in_out_maps();
};


} // namespace Lpm

#endif
