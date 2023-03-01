#ifndef LPM_COMPADRE_HPP
#define LPM_COMPADRE_HPP

#include "Compadre_GMLS.hpp"
#include "Compadre_Operators.hpp"
#include "LpmConfig.h"
#include "lpm_coords.hpp"
#include "lpm_geometry.hpp"

namespace Lpm {
namespace gmls {

struct Params {
  Real eps_multiplier;
  Int samples_order;
  Int manifold_order;
  Real samples_weight_pwr;
  Real manifold_weight_pwr;
  Int ambient_dim;
  Int topo_dim;
  Int min_neighbors;

  std::string info_string(const int tab_lev = 0) const;

  Params()
      : eps_multiplier(2.0),
        samples_order(3),
        manifold_order(3),
        samples_weight_pwr(2.0),
        manifold_weight_pwr(2.0),
        ambient_dim(3),
        topo_dim(2),
        min_neighbors(Compadre::GMLS::getNP(samples_order, topo_dim)) {}

  Params(const Params& other) = default;

  Params(const Int order, const Int dim = 3)
      : eps_multiplier(2.0),
        samples_order(order),
        manifold_order(order),
        samples_weight_pwr(2.0),
        manifold_weight_pwr(2.0),
        ambient_dim(dim),
        topo_dim(2),
        min_neighbors(Compadre::GMLS::getNP(order, topo_dim)) {}
};

struct Neighborhoods {
  using host_crd_view = typename Kokkos::View<Real**>::HostMirror;
  Kokkos::View<Index**> neighbor_lists;
  Kokkos::View<Real*> neighborhood_radii;
  Real r_min;
  Real r_max;
  Int n_min;
  Int n_max;

  Neighborhoods(const host_crd_view host_src_crds,
                const host_crd_view host_tgt_crds, const Params& params);

  KOKKOS_INLINE_FUNCTION
  Real min_radius() const { return r_min; }

  KOKKOS_INLINE_FUNCTION
  Real max_radius() const { return r_max; }

  KOKKOS_INLINE_FUNCTION
  Int min_neighbors() const { return n_min; }

  KOKKOS_INLINE_FUNCTION
  Int max_neighbors() const { return n_max; }

  struct RadiusReducer {
    Kokkos::View<Real*> radii;

    RadiusReducer(const Kokkos::View<Real*> v) : radii(v) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const Index i, Kokkos::MinMaxScalar<Real>& r) const {
      if (radii(i) < r.min_val) r.min_val = radii(i);
      if (radii(i) > r.max_val) r.max_val = radii(i);
    }
  };

  struct NReducer {
    Kokkos::View<Index**> neighbors;

    NReducer(const Kokkos::View<Index**> nl) : neighbors(nl) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const Index i, Kokkos::MinMaxScalar<Int>& mm) const {
      if (neighbors(i, 0) < mm.min_val) mm.min_val = neighbors(i, 0);
      if (neighbors(i, 0) > mm.max_val) mm.max_val = neighbors(i, 0);
    }
  };

  std::string info_string(const int tab_lev = 0) const;
  void compute_bds();
};

template <typename SrcCrdViewType, typename TgtCrdViewType>
Compadre::GMLS plane_scalar_gmls(
    const SrcCrdViewType& src_crds, const TgtCrdViewType& tgt_crds,
    const Neighborhoods& nn, const Params& params,
    const std::vector<Compadre::TargetOperation>& ops) {
  constexpr auto reconstruction_space =
      Compadre::ReconstructionSpace::ScalarTaylorPolynomial;
  constexpr auto problem_type = Compadre::ProblemType::STANDARD;
  constexpr auto solver_type = Compadre::DenseSolverType::QR;
  constexpr auto constraint_type = Compadre::ConstraintType::NO_CONSTRAINT;
  constexpr auto sampling_functional = Compadre::PointSample;
  constexpr auto data_functional = sampling_functional;

  Compadre::GMLS result(reconstruction_space, sampling_functional,
                        data_functional, params.samples_order,
                        PlaneGeometry::ndim, solver_type, problem_type,
                        constraint_type, params.manifold_order);

  result.setProblemData(nn.neighbor_lists, src_crds, tgt_crds,
                        nn.neighborhood_radii);

  result.addTargets(ops);

  constexpr auto weighting_type = Compadre::WeightingFunctionType::Power;
  result.setWeightingType(weighting_type);
  result.setWeightingParameter(params.samples_weight_pwr);

  result.generateAlphas();

  return result;
}

template <typename SrcCrdViewType, typename TgtCrdViewType>
Compadre::GMLS sphere_scalar_gmls(
    const SrcCrdViewType src_crds, const TgtCrdViewType tgt_crds,
    const Neighborhoods& nn, const Params& params,
    const std::vector<Compadre::TargetOperation>& ops) {
  constexpr auto reconstruction_space =
      Compadre::ReconstructionSpace::ScalarTaylorPolynomial;
  constexpr auto problem_type = Compadre::ProblemType::MANIFOLD;
  constexpr auto solver_type = Compadre::DenseSolverType::QR;
  constexpr auto constraint_type = Compadre::ConstraintType::NO_CONSTRAINT;
  constexpr auto sampling_functional = Compadre::PointSample;
  constexpr auto data_functional = sampling_functional;

  Compadre::GMLS result(reconstruction_space, sampling_functional,
                        data_functional, params.samples_order,
                        SphereGeometry::ndim, solver_type, problem_type,
                        constraint_type, params.manifold_order);

  result.setProblemData(nn.neighbor_lists, src_crds, tgt_crds,
                        nn.neighborhood_radii);

  result.addTargets(ops);

  constexpr auto weighting_type = Compadre::WeightingFunctionType::Power;
  result.setWeightingType(weighting_type);
  result.setWeightingParameter(params.samples_weight_pwr);

  constexpr bool use_to_orient = true;
  result.setReferenceOutwardNormalDirection(tgt_crds, use_to_orient);

  result.setCurvatureWeightingType(weighting_type);
  result.setCurvatureWeightingParameter(params.manifold_weight_pwr);

  result.generateAlphas();

  return result;
}

}  // namespace gmls
}  // namespace Lpm

#endif
