#ifndef LPM_COMPADRE_REMESH_HPP
#define LPM_COMPADRE_REMESH_HPP

#include "LpmConfig.h"
#include "lpm_compadre.hpp"
#include "lpm_coriolis.hpp"
#include "lpm_geometry.hpp"
#include "lpm_logger.hpp"
#include "mesh/lpm_gather_mesh_data.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#include "mesh/lpm_refinement.hpp"
#include "mesh/lpm_refinement_flags.hpp"
#include "mesh/lpm_scatter_mesh_data.hpp"


namespace Lpm {

template <typename SeedType>
struct CompadreRemesh {
  using vert_scalar_field_map = std::map<std::string, ScalarField<VertexField>>;
  using vert_vector_field_map = std::map<std::string,
    VectorField<typename SeedType::geo,VertexField>>;
  using face_scalar_field_map = std::map<std::string, ScalarField<FaceField>>;
  using face_vector_field_map = std::map<std::string,
    VectorField<typename SeedType::geo,FaceField>>;
  using host_crd_view = typename SeedType::geo::crd_view_type::HostMirror;
  using CoriolisType = typename std::conditional<
    std::is_same<typename SeedType::geo, PlaneGeometry>::value,
      CoriolisBetaPlane, CoriolisSphere>::type;

  gmls::Params gmls_params;
  gmls::Neighborhoods neighborhoods;

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

  CompadreRemesh(PolyMesh2d<SeedType>& new_mesh,
    vert_scalar_field_map& new_vert_scalars,
    face_scalar_field_map& new_face_scalars,
    vert_vector_field_map& new_vert_vectors,
    face_vector_field_map& new_face_vectors,
    const PolyMesh2d<SeedType>& old_mesh,
    const vert_scalar_field_map& old_vert_scalars,
    const face_scalar_field_map& old_face_scalars,
    const vert_vector_field_map& old_vert_vectors,
    const face_vector_field_map& old_face_vectors,
    const gmls::Params& params,
    const std::shared_ptr<spdlog::logger> logger_in = nullptr);

  void uniform_direct_remesh();

  template <typename VorticityFunctor>
  void uniform_indirect_remesh(const VorticityFunctor& vorticity,
    const CoriolisType& coriolis);

  template <typename VorticityFunctor, typename Tracer1>
  void uniform_indirect_remesh(const VorticityFunctor& vorticity,
    const CoriolisType& coriolis, const Tracer1& tracer1);

  template <typename VorticityFunctor, typename Tracer1, typename Tracer2>
  void uniform_indirect_remesh(const VorticityFunctor& vorticity,
    const CoriolisType& coriolis, const Tracer1& tracer1, const Tracer2& tracer2);

  private:
    Compadre::ReconstructionSpace scalar_reconstruction_space;
    Compadre::ReconstructionSpace vector_reconstruction_space;
    Compadre::ProblemType problem_type;
    Compadre::DenseSolverType solver_type;
    Compadre::SamplingFunctional scalar_sampling_functional;
    Compadre::SamplingFunctional vector_sampling_functional;
    Compadre::WeightingFunctionType weight_type;
    Compadre::ConstraintType constraint_type;

    std::unique_ptr<GatherMeshData<SeedType>> old_gather;
    std::unique_ptr<GatherMeshData<SeedType>> new_gather;
    std::unique_ptr<ScatterMeshData<SeedType>> new_scatter;
    std::unique_ptr<Compadre::GMLS> scalar_gmls;
    std::unique_ptr<Compadre::GMLS> vector_gmls;

    std::vector<Compadre::TargetOperation> scalar_gmls_ops;
    std::vector<Compadre::TargetOperation> vector_gmls_ops;

    void interpolate_lag_crds();

    std::shared_ptr<spdlog::logger> logger;
};

} // namespace Lpm

#endif
