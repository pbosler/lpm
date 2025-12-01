#ifndef LPM_COMPADRE_DFS_REMESH_IMPL_HPP
#define LPM_COMPADRE_DFS_REMESH_IMPL_HPP

#include "LpmConfig.h"
#include "dfs/lpm_compadre_dfs_remesh.hpp"
#include "mesh/lpm_gather_mesh_data_impl.hpp"
#include "mesh/lpm_polymesh2d_impl.hpp"
#include "mesh/lpm_scatter_mesh_data_impl.hpp"
#include "util/lpm_stl_utils.hpp"

#include <Compadre_Evaluator.hpp>
#include <Compadre_GMLS.hpp>
#include <Compadre_Operators.hpp>

namespace Lpm {
namespace DFS {

template <typename SeedType>
std::string CompadreDfsRemesh<SeedType>::info_string(const int tab_lev) const {
  std::ostringstream ss;
  auto tabstr = indent_string(tab_lev);
  ss << tabstr << "CompadreDfsRemesh<" << SeedType::id_string() << "> info:\n";
  tabstr += "\t";
  ss << tabstr << "old_gather: " << old_gather->info_string(tab_lev+1);
  ss << tabstr << "new_gather: " << new_gather->info_string(tab_lev+1);
  ss << tabstr << "----new_vert_scalars:\n";
  for (const auto& sf : new_vert_scalars) {
    ss << sf.first << sf.second.info_string();
  }
  ss << tabstr << "----new_face_scalars:\n";
  for (const auto& sf : new_face_scalars) {
    ss << sf.first << sf.second.info_string();
  }
  ss << tabstr << "----new_grid_scalars:\n";
  for (const auto& sf : new_grid_scalars) {
    ss << sf.first << sf.second.info_string();
  }
  ss << tabstr << "----new_vert_vectors:\n";
  for (const auto& sf : new_vert_vectors) {
    ss << sf.first << sf.second.info_string();
  }
  ss << tabstr << "----new_face_vectors:\n";
  for (const auto& sf : new_face_scalars) {
    ss << sf.first << sf.second.info_string();
  }
  ss << tabstr << "----new_grid_vectors:\n";
  for (const auto& sf : new_grid_vectors) {
    ss << sf.first << sf.second.info_string();
  }
  ss << tabstr << "----old_vert_scalars:\n";
  for (const auto& sf : old_vert_scalars) {
    ss << sf.first << sf.second.info_string();
  }
  ss << tabstr << "----old_face_scalars:\n";
  for (const auto& sf : old_face_scalars) {
    ss << sf.first << sf.second.info_string();
  }
  ss << tabstr << "----old_vert_vectors:\n";
  for (const auto& sf : old_vert_vectors) {
    ss << sf.first << sf.second.info_string();
  }
  ss << tabstr << "----old_face_vectors:\n";
  for (const auto& sf : old_face_vectors) {
    ss << sf.first << sf.second.info_string();
  }
  return ss.str();
}

template <typename SeedType>
CompadreDfsRemesh<SeedType>::CompadreDfsRemesh(PolyMesh2d<SeedType>& new_mesh,
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
    const std::shared_ptr<spdlog::logger> logger_in) :
    new_mesh(new_mesh),
    new_vert_scalars(new_vert_scalars),
    new_face_scalars(new_face_scalars),
    new_grid_scalars(new_grid_scalars),
    new_vert_vectors(new_vert_vectors),
    new_face_vectors(new_face_vectors),
    new_grid_vectors(new_grid_vectors),
    grid_crds(grid_crds),
    old_mesh(old_mesh),
    old_vert_scalars(old_vert_scalars),
    old_face_scalars(old_face_scalars),
    old_vert_vectors(old_vert_vectors),
    old_face_vectors(old_face_vectors),
    gmls_params(params),
    logger(logger_in),
    scalar_reconstruction_space(Compadre::ReconstructionSpace::ScalarTaylorPolynomial),
    vector_reconstruction_space(
      Compadre::ReconstructionSpace::VectorTaylorPolynomial),
    scalar_sampling_functional(Compadre::PointSample),
    vector_sampling_functional(Compadre::ManifoldVectorPointSample ),
    problem_type(Compadre::ProblemType::MANIFOLD),
    solver_type(Compadre::DenseSolverType::QR),
    constraint_type(Compadre::ConstraintType::NO_CONSTRAINT),
    weight_type(Compadre::WeightingFunctionType::Power)
{
    if (!logger_in) {
      logger = lpm_logger();
    }

    old_gather = std::make_unique<GatherMeshData<SeedType>>(old_mesh);
    old_gather->unpack_coordinates();
    old_gather->init_scalar_fields(old_vert_scalars, old_face_scalars);
    old_gather->init_vector_fields(old_vert_vectors, old_face_vectors);
    old_gather->gather_scalar_fields(old_vert_scalars, old_face_scalars);
    old_gather->gather_vector_fields(old_vert_vectors, old_face_vectors);

    new_gather = std::make_unique<GatherMeshData<SeedType>>(new_mesh);
    new_gather->init_scalar_fields(new_vert_scalars, new_face_scalars);
    new_gather->init_vector_fields(new_vert_vectors, new_face_vectors);

    new_scatter = std::make_unique<ScatterMeshData<SeedType>>(*new_gather, new_mesh);

    old_gather->update_host();
    new_gather->update_host();

    mesh_neighborhoods = gmls::Neighborhoods(old_gather->h_phys_crds, new_gather->h_phys_crds,
      gmls_params);
    grid_neighborhoods = gmls::Neighborhoods(old_gather->h_phys_crds, grid_crds.get_const_host_crd_view(), gmls_params);

    const auto scalar_data_functional = scalar_sampling_functional;
    const auto vector_data_functional = vector_sampling_functional;

    gmls_setup();

}

template <typename SeedType>
void CompadreDfsRemesh<SeedType>::interpolate_lag_crds() {
  Compadre::Evaluator scalar_eval(scalar_gmls.get());
  const auto lag_x_src = old_gather->lag_x;
  const auto lag_y_src = old_gather->lag_y;
  const auto lag_z_src = old_gather->lag_z;

  const auto lag_x_dst =
    scalar_eval.applyAlphasToDataAllComponentsAllTargetSites<Real*, DevMemory>(
      lag_x_src, Compadre::ScalarPointEvaluation, Compadre::PointSample);
  const auto lag_y_dst =
    scalar_eval.applyAlphasToDataAllComponentsAllTargetSites<Real*, DevMemory>(
      lag_y_src, Compadre::ScalarPointEvaluation, Compadre::PointSample);
  const auto lag_z_dst =
    scalar_eval.applyAlphasToDataAllComponentsAllTargetSites<Real*, DevMemory>(
      lag_z_src, Compadre::ScalarPointEvaluation, Compadre::PointSample);

  auto lag_crds_dst = new_gather->lag_crds;
  Kokkos::parallel_for(new_gather->n(),
    KOKKOS_LAMBDA (const Index i) {
      lag_crds_dst(i,0) = lag_x_dst(i);
      lag_crds_dst(i,1) = lag_y_dst(i);
      lag_crds_dst(i,2) = lag_z_dst(i);
      auto lci = Kokkos::subview(lag_crds_dst, i, Kokkos::ALL);
      SphereGeometry::normalize(lci);
    });
  }

template <typename SeedType>
void CompadreDfsRemesh<SeedType>::uniform_direct_remesh() {
  direct_remesh();
}

template <typename SeedType>
void CompadreDfsRemesh<SeedType>::direct_remesh() {
  interpolate_lag_crds();
  Compadre::Evaluator scalar_eval(scalar_gmls.get());

  for (const auto& scalar_field : old_gather->scalar_fields) {
    const auto src_data = scalar_field.second;
    new_gather->scalar_fields.at(scalar_field.first) =
      scalar_eval.applyAlphasToDataAllComponentsAllTargetSites<Real*,DevMemory>(
        src_data, Compadre::ScalarPointEvaluation, Compadre::PointSample);
  }


  Compadre::Evaluator vector_eval(vector_gmls.get());
  for (const auto& vector_field : old_gather->vector_fields) {
    const auto src_data = vector_field.second;
    new_gather->vector_fields.at(vector_field.first) =
    vector_eval.applyAlphasToDataAllComponentsAllTargetSites<Real**,DevMemory>(
      src_data,
      Compadre::VectorPointEvaluation,
      Compadre::ManifoldVectorPointSample);
  }

  new_scatter->scatter_lag_crds();
  new_scatter->scatter_fields(new_vert_scalars, new_face_scalars,
    new_vert_vectors, new_face_vectors);

  Compadre::Evaluator grid_scalar_eval(grid_scalar_gmls.get());
  for (auto& scalar_field : new_grid_scalars) {
    const auto src_data = old_gather->scalar_fields.at(scalar_field.first);
    const auto new_view =
      grid_scalar_eval.applyAlphasToDataAllComponentsAllTargetSites<Real*, DevMemory>(
        src_data, Compadre::ScalarPointEvaluation, Compadre::PointSample);

//     Real new_max;
//     Kokkos::parallel_reduce(new_view.extent(0),
//       KOKKOS_LAMBDA (const Index i, Real& m) {
//         if (abs(new_view(i)) > m ) m = abs(new_view(i));
//       }, Kokkos::Max<Real>(new_max));
//     Real src_max;
//     Kokkos::parallel_reduce(src_data.extent(0),
//       KOKKOS_LAMBDA (const Index i, Real& m) {
//         if (abs(src_data(i)) > m ) m = abs(src_data(i));
//       }, Kokkos::Max<Real>(src_max));

    Kokkos::deep_copy(scalar_field.second.view, new_view);
  }

  Compadre::Evaluator grid_vector_eval(grid_vector_gmls.get());
  for (auto& vector_field : new_grid_vectors) {
    const auto src_data = old_gather->vector_fields.at(vector_field.first);
    const auto new_view =
      grid_vector_eval.applyAlphasToDataAllComponentsAllTargetSites<Real**, DevMemory>(
        src_data,
        Compadre::VectorPointEvaluation,
        Compadre::ManifoldVectorPointSample);

    Kokkos::deep_copy(vector_field.second.view, new_view);
  }

  logger->debug("direct remesh complete");
}

template <typename SeedType> template <typename FlagType>
void CompadreDfsRemesh<SeedType>::adaptive_direct_remesh(Refinement<SeedType>& refiner,
  const FlagType& flag) {

  uniform_direct_remesh();
  logger->debug("adaptive remesh ftle info: {}", new_face_scalars.at("ftle").info_string());
  Index face_start_idx = 0;
  for (int amr_i = 0; amr_i < new_mesh.params.amr_limit; ++amr_i) {
    Index face_end_idx = new_mesh.n_faces_host();

    refiner.iterate(face_start_idx, face_end_idx, flag);
    if (refiner.total() > 0) {

      new_mesh.divide_flagged_faces(refiner.flags, *logger);

      new_gather.reset(new GatherMeshData<SeedType>(new_mesh));
      new_gather->init_scalar_fields(new_vert_scalars, new_face_scalars);
      new_gather->init_vector_fields(new_vert_vectors, new_face_vectors);
      new_gather->update_host();

      mesh_neighborhoods = gmls::Neighborhoods(old_gather->h_phys_crds, new_gather->h_phys_crds,
        gmls_params);

      new_scatter.reset(new ScatterMeshData<SeedType>(*new_gather, new_mesh));

      gmls_setup();

      direct_remesh();

      face_start_idx = face_end_idx;
    }
    else {
      break;
    }
  }
}

template <typename SeedType> template <typename FlagType1, typename FlagType2>
void CompadreDfsRemesh<SeedType>::adaptive_direct_remesh(Refinement<SeedType>& refiner,
      const FlagType1& flag1, const FlagType2& flag2) {

  uniform_direct_remesh();
  logger->debug("adaptive remesh ftle info: {}", new_face_scalars.at("ftle").info_string());
  Index face_start_idx = 0;
  for (int amr_i = 0; amr_i < new_mesh.params.amr_limit; ++amr_i) {
    Index face_end_idx = new_mesh.n_faces_host();

    refiner.iterate(face_start_idx, face_end_idx, flag1, flag2);
    logger->debug("adaptive remesh : {}", refiner.info_string());
    if (refiner.total() > 0) {

      new_mesh.divide_flagged_faces(refiner.flags, *logger);

      new_gather.reset(new GatherMeshData<SeedType>(new_mesh));
      new_gather->init_scalar_fields(new_vert_scalars, new_face_scalars);
      new_gather->init_vector_fields(new_vert_vectors, new_face_vectors);
      new_gather->update_host();

      mesh_neighborhoods = gmls::Neighborhoods(old_gather->h_phys_crds, new_gather->h_phys_crds,
        gmls_params);

      new_scatter.reset(new ScatterMeshData<SeedType>(*new_gather, new_mesh));

      gmls_setup();

      direct_remesh();

      face_start_idx = face_end_idx;
    }
    else {
      break;
    }
  }
}

template <typename SeedType>
void CompadreDfsRemesh<SeedType>::gmls_setup() {
  const auto scalar_data_functional = scalar_sampling_functional;
  const auto vector_data_functional = vector_sampling_functional;
  if (!scalar_gmls) {
    scalar_gmls = std::make_unique<Compadre::GMLS>(
      scalar_reconstruction_space,
      scalar_sampling_functional,
      scalar_data_functional,
      gmls_params.samples_order,
      SeedType::geo::ndim,
      solver_type,
      problem_type,
      constraint_type,
      gmls_params.manifold_order);

    vector_gmls = std::make_unique<Compadre::GMLS>(
      vector_reconstruction_space,
      vector_sampling_functional,
      vector_data_functional,
      gmls_params.samples_order,
      SeedType::geo::ndim,
      solver_type,
      problem_type,
      constraint_type,
      gmls_params.manifold_order);

    grid_scalar_gmls = std::make_unique<Compadre::GMLS>(
      scalar_reconstruction_space,
      scalar_sampling_functional,
      scalar_data_functional,
      gmls_params.samples_order,
      SeedType::geo::ndim,
      solver_type,
      problem_type,
      constraint_type,
      gmls_params.manifold_order);

    grid_vector_gmls = std::make_unique<Compadre::GMLS>(
      vector_reconstruction_space,
      vector_sampling_functional,
      vector_data_functional,
      gmls_params.samples_order,
      SeedType::geo::ndim,
      solver_type,
      problem_type,
      constraint_type,
      gmls_params.manifold_order);

    scalar_gmls_ops = {Compadre::ScalarPointEvaluation,
        Compadre::LaplacianOfScalarPointEvaluation};
    vector_gmls_ops = {Compadre::VectorPointEvaluation};
  }
  else {
    scalar_gmls.reset(new Compadre::GMLS(
      scalar_reconstruction_space,
      scalar_sampling_functional,
      scalar_data_functional,
      gmls_params.samples_order,
      SeedType::geo::ndim,
      solver_type,
      problem_type,
      constraint_type,
      gmls_params.manifold_order));

    vector_gmls.reset(new Compadre::GMLS(
      vector_reconstruction_space,
      vector_sampling_functional,
      vector_data_functional,
      gmls_params.samples_order,
      SeedType::geo::ndim,
      solver_type,
      problem_type,
      constraint_type,
      gmls_params.manifold_order));

    grid_scalar_gmls.reset(new Compadre::GMLS(
      scalar_reconstruction_space,
      scalar_sampling_functional,
      scalar_data_functional,
      gmls_params.samples_order,
      SeedType::geo::ndim,
      solver_type,
      problem_type,
      constraint_type,
      gmls_params.manifold_order));

    grid_vector_gmls.reset(new Compadre::GMLS(
      vector_reconstruction_space,
      vector_sampling_functional,
      vector_data_functional,
      gmls_params.samples_order,
      SeedType::geo::ndim,
      solver_type,
      problem_type,
      constraint_type,
      gmls_params.manifold_order));
  }

    scalar_gmls->setProblemData(mesh_neighborhoods.neighbor_lists,
      old_gather->phys_crds, new_gather->phys_crds,
      mesh_neighborhoods.neighborhood_radii);
    scalar_gmls->addTargets(scalar_gmls_ops);
    scalar_gmls->setWeightingType(weight_type);
    scalar_gmls->setWeightingParameter(gmls_params.samples_weight_pwr);

    vector_gmls->setProblemData(mesh_neighborhoods.neighbor_lists,
      old_gather->phys_crds, new_gather->phys_crds,
      mesh_neighborhoods.neighborhood_radii);
    vector_gmls->addTargets(vector_gmls_ops);
    vector_gmls->setWeightingType(weight_type);
    vector_gmls->setWeightingParameter(gmls_params.samples_weight_pwr);

    grid_scalar_gmls->setProblemData(grid_neighborhoods.neighbor_lists,
      old_gather->phys_crds, grid_crds.view,
      grid_neighborhoods.neighborhood_radii);
    grid_scalar_gmls->addTargets(scalar_gmls_ops);
    grid_scalar_gmls->setWeightingType(weight_type);
    grid_scalar_gmls->setWeightingParameter(gmls_params.samples_weight_pwr);

    grid_vector_gmls->setProblemData(grid_neighborhoods.neighbor_lists,
      old_gather->phys_crds, grid_crds.view,
      grid_neighborhoods.neighborhood_radii);
    grid_vector_gmls->addTargets(vector_gmls_ops);
    grid_vector_gmls->setWeightingType(weight_type);
    grid_vector_gmls->setWeightingParameter(gmls_params.samples_weight_pwr);

    constexpr bool use_to_orient = true;
    scalar_gmls->setReferenceOutwardNormalDirection(
      new_gather->phys_crds, use_to_orient);
    scalar_gmls->setCurvatureWeightingType(weight_type);
    scalar_gmls->setCurvatureWeightingParameter(gmls_params.manifold_weight_pwr);

    vector_gmls->setReferenceOutwardNormalDirection(
      new_gather->phys_crds, use_to_orient);
    vector_gmls->setCurvatureWeightingType(weight_type);
    vector_gmls->setCurvatureWeightingParameter(gmls_params.manifold_weight_pwr);

    grid_scalar_gmls->setReferenceOutwardNormalDirection(
      grid_crds.view, use_to_orient);
    grid_scalar_gmls->setCurvatureWeightingType(weight_type);
    grid_scalar_gmls->setCurvatureWeightingParameter(gmls_params.manifold_weight_pwr);

    grid_vector_gmls->setReferenceOutwardNormalDirection(
      grid_crds.view, use_to_orient);
    grid_vector_gmls->setCurvatureWeightingType(weight_type);
    grid_vector_gmls->setCurvatureWeightingParameter(gmls_params.manifold_weight_pwr);

    scalar_gmls->generateAlphas();
    vector_gmls->generateAlphas();
    grid_scalar_gmls->generateAlphas();
    grid_vector_gmls->generateAlphas();
}

} // namespace DFS
} // namespace Lpm
#endif
