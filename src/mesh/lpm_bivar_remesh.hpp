#ifndef LPM_BIVAR_REMESH_HPP
#define LPM_BIVAR_REMESH_HPP

#include "LpmConfig.h"
#include "lpm_coriolis.hpp"
#include "lpm_geometry.hpp"
#include "fortran/lpm_bivar_interface.hpp"
#include "fortran/lpm_bivar_interface_impl.hpp"
#include "mesh/lpm_gather_mesh_data.hpp"
#include "mesh/lpm_gather_mesh_data_impl.hpp"
#include "mesh/lpm_refinement.hpp"
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
     const face_vector_field_map& old_face_vectors,
     const std::shared_ptr<spdlog::logger> login = nullptr);

  /** Directly interpolate all data from old mesh to new mesh without
    any spatial refinement.
  */
  void uniform_direct_remesh();

  /** Interpolate data from old mesh to new mesh, without any
    spatial refinement.
    Use indirect interpolation to define Lagrangian data.

    @param [in] functor_map <key, value> pairs where keys are std::strings
      naming Lagrangian scalar fields, and the values are the functors
      that define their initial data (e.g., from lpm_tracer_gallery.hpp
      or lpm_vorticity_gallery.hpp).
  */
  template <typename VorticityFunctor>
  void uniform_indirect_remesh(const VorticityFunctor& vorticity,
    const CoriolisBetaPlane& coriolis);

  template <typename VorticityFunctor, typename Tracer1>
  void uniform_indirect_remesh(const VorticityFunctor& vorticity,
    const CoriolisBetaPlane& coriolis, const Tracer1& tracer1);

  template <typename VorticityFunctor, typename Tracer1, typename Tracer2>
  void uniform_indirect_remesh(const VorticityFunctor& vorticity,
    const CoriolisBetaPlane& coriolis, const Tracer1& tracer1, const Tracer2& tracer2);

  template <typename FlagType>
  void adaptive_direct_remesh(Refinement<SeedType>& refiner, const FlagType& flag);

  template <typename FlagType1, typename FlagType2>
  void adaptive_direct_remesh(Refinement<SeedType>& refiner,
    const FlagType1& flag1, const FlagType2& flag2);

  template <typename FlagType1, typename FlagType2, typename FlagType3>
  void adaptive_direct_remesh(Refinement<SeedType>& refiner,
    const FlagType1& flag1, const FlagType2& flag2, const FlagType3& flag3);

  template <typename VorticityFunctor, typename RefinerType, typename FlagType>
  void adaptive_indirect_remesh(const VorticityFunctor& vorticity,
    const CoriolisBetaPlane& coriolis,
    RefinerType& refiner, const FlagType& flag);

  template <typename VorticityFunctor, typename RefinerType,
    typename FlagType1, typename FlagType2>
  void adaptive_indirect_remesh(const VorticityFunctor& vorticity,
    const CoriolisBetaPlane& coriolis,
    RefinerType& refiner, const FlagType1& flag1, const FlagType2& flag);

  template <typename VorticityFunctor, typename RefinerType,
    typename FlagType1, typename FlagType2, typename FlagType3>
  void adaptive_indirect_remesh(const VorticityFunctor& vorticity,
    const CoriolisBetaPlane& coriolis,
    RefinerType& refiner, const FlagType1& flag1, const FlagType2& flag,
    const FlagType3& flag3);

  protected:
    std::unique_ptr<BivarInterface<SeedType>> bivar;
    std::unique_ptr<GatherMeshData<SeedType>> old_gather;
    std::unique_ptr<GatherMeshData<SeedType>> new_gather;
    std::unique_ptr<ScatterMeshData<SeedType>> new_scatter;

    in_out_map scalar_in_out_map;
    in_out_map vector_in_out_map;

    void build_in_out_maps();

    std::shared_ptr<spdlog::logger> logger;
};


} // namespace Lpm

#endif
