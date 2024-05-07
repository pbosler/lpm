#ifndef LPM_COMPADRE_REMESH_IMPL_HPP
#define LPM_COMPADRE_REMESH_IMPL_HPP

#include "LpmConfig.h"
#include "mesh/lpm_compadre_remesh.hpp"
#include "mesh/lpm_gather_mesh_data_impl.hpp"
#include "mesh/lpm_refinement.hpp"
#include "mesh/lpm_refinement_flags.hpp"
#include "mesh/lpm_scatter_mesh_data_impl.hpp"
#include "util/lpm_stl_utils.hpp"

#include <Compadre_Evaluator.hpp>
#include <Compadre_GMLS.hpp>
#include <Compadre_Operators.hpp>

namespace Lpm {

template <typename SeedType>
CompadreRemesh<SeedType>::CompadreRemesh(PolyMesh2d<SeedType>& new_mesh,
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
    const std::shared_ptr<spdlog::logger> logger_in) :
    new_mesh(new_mesh),
    new_vert_scalars(new_vert_scalars),
    new_face_scalars(new_face_scalars),
    new_vert_vectors(new_vert_vectors),
    new_face_vectors(new_face_vectors),
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
    vector_sampling_functional( (std::is_same<typename SeedType::geo, SphereGeometry>::value ? Compadre::ManifoldVectorPointSample :  Compadre::VectorPointSample) ),
    problem_type((std::is_same<typename SeedType::geo, SphereGeometry>::value ?
      Compadre::ProblemType::MANIFOLD : Compadre::ProblemType::STANDARD)),
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
    neighborhoods = gmls::Neighborhoods(old_gather->h_phys_crds, new_gather->h_phys_crds,
      gmls_params);

    const auto scalar_data_functional = scalar_sampling_functional;
    const auto vector_data_functional = vector_sampling_functional;
    scalar_gmls = std::make_unique<Compadre::GMLS>(
      scalar_reconstruction_space,
      scalar_sampling_functional,
      scalar_data_functional,
      params.samples_order,
      SeedType::geo::ndim,
      solver_type,
      problem_type,
      constraint_type,
      params.manifold_order);

    vector_gmls = std::make_unique<Compadre::GMLS>(
      vector_reconstruction_space,
      vector_sampling_functional,
      vector_data_functional,
      params.samples_order,
      SeedType::geo::ndim,
      solver_type,
      problem_type,
      constraint_type,
      params.manifold_order);

    scalar_gmls_ops = {Compadre::ScalarPointEvaluation,
        Compadre::LaplacianOfScalarPointEvaluation};
    vector_gmls_ops = {Compadre::VectorPointEvaluation};

    scalar_gmls->setProblemData(neighborhoods.neighbor_lists,
      old_gather->phys_crds, new_gather->phys_crds,
      neighborhoods.neighborhood_radii);
    scalar_gmls->addTargets(scalar_gmls_ops);

    vector_gmls->setProblemData(neighborhoods.neighbor_lists,
      old_gather->phys_crds, new_gather->phys_crds,
      neighborhoods.neighborhood_radii);
    vector_gmls->addTargets(vector_gmls_ops);

    scalar_gmls->setWeightingType(weight_type);
    scalar_gmls->setWeightingParameter(params.samples_weight_pwr);

    vector_gmls->setWeightingType(weight_type);
    vector_gmls->setWeightingParameter(params.samples_weight_pwr);

    if constexpr (std::is_same<typename SeedType::geo,SphereGeometry>::value) {
      constexpr bool use_to_orient = true;
      scalar_gmls->setReferenceOutwardNormalDirection(
        new_gather->phys_crds, use_to_orient);
      scalar_gmls->setCurvatureWeightingType(weight_type);
      scalar_gmls->setCurvatureWeightingParameter(params.manifold_weight_pwr);

      vector_gmls->setReferenceOutwardNormalDirection(
        new_gather->phys_crds, use_to_orient);
      vector_gmls->setCurvatureWeightingType(weight_type);
      vector_gmls->setCurvatureWeightingParameter(params.manifold_weight_pwr);
    }

    scalar_gmls->generateAlphas();
    vector_gmls->generateAlphas();
    logger->debug("old_gather info: {}", old_gather->info_string());
    logger->debug("new_gather info: {}", new_gather->info_string());
}

template <typename SeedType>
void CompadreRemesh<SeedType>::uniform_direct_remesh() {
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
    });

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
      (std::is_same<typename SeedType::geo, SphereGeometry>::value ?
      Compadre::ManifoldVectorPointSample : Compadre::VectorPointSample));
  }

  new_scatter->scatter_lag_crds();
  new_scatter->scatter_fields(new_vert_scalars, new_face_scalars,
    new_vert_vectors, new_face_vectors);

  logger->debug("uniform direct remesh complete");
}

template <typename SeedType> template <typename VorticityFunctor>
void CompadreRemesh<SeedType>::uniform_indirect_remesh(const VorticityFunctor& vorticity,
    const CoriolisType& coriolis) {

  Compadre::Evaluator vector_eval(vector_gmls.get());
  const auto lag_src = old_gather->lag_crds;

  new_gather->lag_crds =
    vector_eval.applyAlphasToDataAllComponentsAllTargetSites<Real**,DevMemory>(
      lag_src, Compadre::VectorPointEvaluation,
      (std::is_same<typename SeedType::geo, SphereGeometry>::value ?
        Compadre::ManifoldVectorPointSample : Compadre::VectorPointSample));

  auto lag_crds = new_gather->lag_crds;
  auto phys_crds = new_gather->phys_crds;
  auto abs_vort = new_gather->scalar_fields.at("absolute_vorticity");
  auto rel_vort = new_gather->scalar_fields.at("relative_vorticity");
  Kokkos::parallel_for(new_gather->n(),
    KOKKOS_LAMBDA (const Index i) {
      const auto xi = Kokkos::subview(phys_crds, i, Kokkos::ALL);
      const auto ai = Kokkos::subview(lag_crds, i, Kokkos::ALL);
      const Real omega = vorticity(ai) + coriolis.f(ai);
      abs_vort(i) = omega;
      rel_vort(i) = omega - coriolis.f(xi);
    });

  new_scatter->scatter_lag_crds();
  new_scatter->scatter_fields(new_vert_scalars, new_face_scalars,
    new_vert_vectors, new_face_vectors);

  logger->debug("uniform indirect remesh complete");
}

template <typename SeedType> template <typename VorticityFunctor, typename Tracer1>
void CompadreRemesh<SeedType>::uniform_indirect_remesh(const VorticityFunctor& vorticity,
    const CoriolisType& coriolis, const Tracer1& tracer1) {

  Compadre::Evaluator vector_eval(vector_gmls.get());
  const auto lag_src = old_gather->lag_crds;

  new_gather->lag_crds =
    vector_eval.applyAlphasToDataAllComponentsAllTargetSites<Real**,DevMemory>(
      lag_src, Compadre::VectorPointEvaluation,
      (std::is_same<typename SeedType::geo, SphereGeometry>::value ?
        Compadre::ManifoldVectorPointSample : Compadre::VectorPointSample));

  auto lag_crds = new_gather->lag_crds;
  auto phys_crds = new_gather->phys_crds;
  auto abs_vort = new_gather->scalar_fields.at("absolute_vorticity");
  auto rel_vort = new_gather->scalar_fields.at("relative_vorticity");
  auto tracer1_view = new_gather->scalar_fields.at(tracer1.name());
  Kokkos::parallel_for(new_gather->n(),
    KOKKOS_LAMBDA (const Index i) {
      const auto xi = Kokkos::subview(phys_crds, i, Kokkos::ALL);
      const auto ai = Kokkos::subview(lag_crds, i, Kokkos::ALL);
      const Real omega = vorticity(ai) + coriolis.f(ai);
      abs_vort(i) = omega;
      rel_vort(i) = omega - coriolis.f(xi);
      tracer1_view(i) = tracer1(ai);
    });

  new_scatter->scatter_lag_crds();
  new_scatter->scatter_fields(new_vert_scalars, new_face_scalars,
    new_vert_vectors, new_face_vectors);

  logger->debug("uniform indirect remesh complete");
}

template <typename SeedType> template <typename VorticityFunctor, typename Tracer1, typename Tracer2>
void CompadreRemesh<SeedType>::uniform_indirect_remesh(const VorticityFunctor& vorticity,
    const CoriolisType& coriolis, const Tracer1& tracer1, const Tracer2& tracer2) {

  Compadre::Evaluator vector_eval(vector_gmls.get());
  const auto lag_src = old_gather->lag_crds;

  new_gather->lag_crds =
    vector_eval.applyAlphasToDataAllComponentsAllTargetSites<Real**,DevMemory>(
      lag_src, Compadre::VectorPointEvaluation,
      (std::is_same<typename SeedType::geo, SphereGeometry>::value ?
        Compadre::ManifoldVectorPointSample : Compadre::VectorPointSample));

  auto lag_crds = new_gather->lag_crds;
  auto phys_crds = new_gather->phys_crds;
  auto abs_vort = new_gather->scalar_fields.at("absolute_vorticity");
  auto rel_vort = new_gather->scalar_fields.at("relative_vorticity");
  auto tracer1_view = new_gather->scalar_fields.at(tracer1.name());
  auto tracer2_view = new_gather->scalar_fields.at(tracer2.name());
  Kokkos::parallel_for(new_gather->n(),
    KOKKOS_LAMBDA (const Index i) {
      const auto xi = Kokkos::subview(phys_crds, i, Kokkos::ALL);
      const auto ai = Kokkos::subview(lag_crds, i, Kokkos::ALL);
      const Real omega = vorticity(ai) + coriolis.f(ai);
      abs_vort(i) = omega;
      rel_vort(i) = omega - coriolis.f(xi);
      tracer1_view(i) = tracer1(ai);
      tracer2_view(i) = tracer2(ai);
    });

  new_scatter->scatter_lag_crds();
  new_scatter->scatter_fields(new_vert_scalars, new_face_scalars,
    new_vert_vectors, new_face_vectors);

  logger->debug("uniform indirect remesh complete");
}

} // namespace Lpm

#endif
