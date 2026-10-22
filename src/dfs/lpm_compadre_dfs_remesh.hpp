#ifndef LPM_COMPADRE_DFS_REMESH_HPP
#define LPM_COMPADRE_DFS_REMESH_HPP

#include "LpmConfig.h"
#include "lpm_compadre.hpp"
#include "lpm_coriolis.hpp"
#include "lpm_geometry.hpp"
#include "lpm_logger.hpp"
#include "dfs/lpm_dfs_grid.hpp"
#include "mesh/lpm_gather_mesh_data.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#include "mesh/lpm_refinement.hpp"
#include "mesh/lpm_refinement_flags.hpp"
#include "mesh/lpm_scatter_mesh_data.hpp"


namespace Lpm {
namespace DFS {

template <typename SeedType>
struct CompadreDfsRemesh {
  using vert_scalar_field_map = std::map<std::string, ScalarField<VertexField>>;
  using vert_vector_field_map = std::map<std::string, VectorField<SphereGeometry,VertexField>>;
  using face_scalar_field_map = std::map<std::string, ScalarField<FaceField>>;
  using face_vector_field_map = std::map<std::string, VectorField<SphereGeometry, FaceField>>;
  using grid_scalar_field_map = std::map<std::string, Lpm::DFS::scalar_field_type>;
  using grid_vector_field_map = std::map<std::string, Lpm::DFS::vector_field_type>;
  using host_crd_view = typename SphereGeometry::crd_view_type::HostMirror;

  gmls::Params gmls_params;
  gmls::Neighborhoods mesh_neighborhoods;
  gmls::Neighborhoods grid_neighborhoods;

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

  const Coords<SphereGeometry>& grid_crds;
  grid_scalar_field_map new_grid_scalars;
  grid_vector_field_map new_grid_vectors;

  CompadreDfsRemesh(PolyMesh2d<SeedType>& new_mesh,
    vert_scalar_field_map& new_vert_scalars,
    face_scalar_field_map& new_face_scalars,
    grid_scalar_field_map& new_grid_scalars,
    vert_vector_field_map& new_vert_vectors,
    face_vector_field_map& new_face_vectors,
    grid_vector_field_map& new_grid_vectors,
    const Coords<SphereGeometry>& grid_crds,
    const PolyMesh2d<SeedType>& old_mesh,
    const vert_scalar_field_map& old_vert_scalars,
    const face_scalar_field_map& old_face_scalars,
    const vert_vector_field_map& old_vert_vectors,
    const face_vector_field_map& old_face_vectors,
    const gmls::Params& params,
    const std::shared_ptr<spdlog::logger> logger_in = nullptr);

  void direct_remesh();
  void uniform_direct_remesh();

  template <typename FlagType>
  void adaptive_direct_remesh(Refinement<SeedType>& refiner, const FlagType& flag);

  template <typename FlagType1, typename FlagType2>
  void adaptive_direct_remesh(Refinement<SeedType>& refiner, const FlagType1& flag1, const FlagType2& flag2);

  std::string info_string(const int tab_lev=0) const;

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
    std::unique_ptr<Compadre::GMLS> grid_scalar_gmls;
    std::unique_ptr<Compadre::GMLS> grid_vector_gmls;

    std::vector<Compadre::TargetOperation> scalar_gmls_ops;
    std::vector<Compadre::TargetOperation> vector_gmls_ops;

    void interpolate_lag_crds();

    std::shared_ptr<spdlog::logger> logger;

    void gmls_setup();
};

} // namespace DFS
} // namespace Lpm
#endif
