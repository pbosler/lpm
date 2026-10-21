#include "lpm_compadre.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>

#include "Compadre_PointCloudSearch.hpp"
#include "util/lpm_string_util.hpp"

namespace Lpm {
namespace gmls {

std::string Params::info_string(const int tab_lev) const {
  auto tab_str = indent_string(tab_lev);
  std::ostringstream ss;
  ss << tab_str << "gmls::Params info:\n";
  tab_str += "\t";
  ss << tab_str << "eps_multiplier = " << eps_multiplier << "\n"
     << tab_str << "samples_order = " << samples_order << "\n"
     << tab_str << "manifold_order = " << manifold_order << "\n"
     << tab_str << "samples_weight_pwr = " << samples_weight_pwr << "\n"
     << tab_str << "manifold_weight_pwr = " << manifold_weight_pwr << "\n"
     << tab_str << "ambient_dim = " << ambient_dim << "\n"
     << tab_str << "topo_dim = " << topo_dim << "\n"
     << tab_str << "min_neighbors = " << min_neighbors << "\n";
  return ss.str();
}

Neighborhoods::Neighborhoods(const host_crd_view host_src_crds,
                             const host_crd_view host_tgt_crds,
                             const Params& params) {
  auto point_cloud_search = Compadre::PointCloudSearch(host_src_crds);
  /**

    Compadre's neighbor list size estimator assumes that points are
    distributed quasi-uniformly.

    todo: create issue/PR in compadre to use dry_run for this.

  */
  Int est_max_neighbors =
      point_cloud_search.getEstimatedNumberNeighborsUpperBound(
          params.min_neighbors, params.topo_dim, params.eps_multiplier);
  neighbor_lists = Kokkos::View<Index**>(
      "neighbor_lists", host_tgt_crds.extent(0), 4 * est_max_neighbors);
  h_neighbors = Kokkos::create_mirror_view(neighbor_lists);

  neighborhood_radii =
      Kokkos::View<Real*>("neighborhood_radii", host_tgt_crds.extent(0));
  h_radii             = Kokkos::create_mirror_view(neighborhood_radii);
  Index max_neighbors = 0;

  bool dry_run = true;
  point_cloud_search.generate2DNeighborListsFromKNNSearch(
      dry_run, host_tgt_crds, h_neighbors, h_radii, params.min_neighbors,
      params.eps_multiplier);

  /// todo: this can be a parallel_reduce on host
  Index max_n = 0;
  for (Index i = 0; i < h_neighbors.extent(0); ++i) {
    if (h_neighbors(i, 0) > max_n) max_n = h_neighbors(i, 0);
  }
  Kokkos::resize(neighbor_lists, host_tgt_crds.extent(0), max_n + 1);
  Kokkos::resize(h_neighbors, host_tgt_crds.extent(0), max_n + 1);

  /// with allocation size determined, we can do the real neighbor search
  dry_run = false;
  point_cloud_search.generate2DNeighborListsFromKNNSearch(
      dry_run, host_tgt_crds, h_neighbors, h_radii, params.min_neighbors,
      params.eps_multiplier);

  Kokkos::deep_copy(neighbor_lists, h_neighbors);
  Kokkos::deep_copy(neighborhood_radii, h_radii);
  compute_bds();
}

Neighborhoods::Neighborhoods(const host_crd_view host_colloc_src_tgt_crds,
                             const Params& params) {
  auto point_cloud_search =
      Compadre::PointCloudSearch(host_colloc_src_tgt_crds);

  Int est_max_neighbors =
      point_cloud_search.getEstimatedNumberNeighborsUpperBound(
          params.min_neighbors, params.topo_dim, params.eps_multiplier);

  neighbor_lists = Kokkos::View<Index**>("neighbor_lists",
                                         host_colloc_src_tgt_crds.extent(0),
                                         4 * est_max_neighbors);
  h_neighbors    = Kokkos::create_mirror_view(neighbor_lists);

  neighborhood_radii  = Kokkos::View<Real*>("neighborhood_radii",
                                            host_colloc_src_tgt_crds.extent(0));
  h_radii             = Kokkos::create_mirror_view(neighborhood_radii);
  Index max_neighbors = 0;

  bool dry_run = true;
  point_cloud_search.generate2DNeighborListsFromKNNSearch(
      dry_run, host_colloc_src_tgt_crds, h_neighbors, h_radii,
      params.min_neighbors, params.eps_multiplier);

  /// todo: this can be a parallel_reduce on host
  Index max_n = 0;
  for (Index i = 0; i < h_neighbors.extent(0); ++i) {
    if (h_neighbors(i, 0) > max_n) max_n = h_neighbors(i, 0);
  }
  Kokkos::resize(neighbor_lists, host_colloc_src_tgt_crds.extent(0), max_n + 1);
  Kokkos::resize(h_neighbors, host_colloc_src_tgt_crds.extent(0), max_n + 1);

  dry_run = false;
  point_cloud_search.generate2DNeighborListsFromKNNSearch(
      dry_run, host_colloc_src_tgt_crds, h_neighbors, h_radii,
      params.min_neighbors, params.eps_multiplier);

  Kokkos::deep_copy(neighbor_lists, h_neighbors);
  Kokkos::deep_copy(neighborhood_radii, h_radii);
  compute_bds();
}

void Neighborhoods::update_neighbors(const host_crd_view host_src_crds,
                                     const host_crd_view host_tgt_crds,
                                     const Params& params, const bool verbose) {
  auto point_cloud_search = Compadre::PointCloudSearch(host_src_crds);
  const bool dry_run      = false;
  point_cloud_search.generate2DNeighborListsFromKNNSearch(
      dry_run, host_tgt_crds, h_neighbors, h_radii, params.min_neighbors,
      params.eps_multiplier);
  if (verbose) {
    std::cout << info_string();
  }
  Kokkos::deep_copy(neighbor_lists, h_neighbors);
  Kokkos::deep_copy(neighborhood_radii, h_radii);
}

void Neighborhoods::compute_bds() {
  Kokkos::MinMaxScalar<Real> rr;
  Kokkos::MinMaxScalar<Int> nn;
  Kokkos::parallel_reduce("Neighborhood reduction (n)",
                          neighbor_lists.extent(0), NReducer(neighbor_lists),
                          Kokkos::MinMax<Int>(nn));
  Kokkos::parallel_reduce(
      "Neighborhood reduction (r)", neighbor_lists.extent(0),
      RadiusReducer(neighborhood_radii), Kokkos::MinMax<Real>(rr));

  r_min = rr.min_val;
  r_max = rr.max_val;
  n_min = nn.min_val;
  n_max = nn.max_val;

  LPM_ASSERT(r_min > 0);
  LPM_ASSERT(n_min > 0);
}

void Neighborhoods::update_neighbors(
    const host_crd_view host_colloc_src_tgt_crds, const Params& params,
    const bool verbose) {
  auto point_cloud_search =
      Compadre::PointCloudSearch(host_colloc_src_tgt_crds);
  const bool dry_run = false;
  point_cloud_search.generate2DNeighborListsFromKNNSearch(
      dry_run, host_colloc_src_tgt_crds, h_neighbors, h_radii,
      params.min_neighbors, params.eps_multiplier);
  if (verbose) {
    std::cout << info_string();
  }
  Kokkos::deep_copy(neighbor_lists, h_neighbors);
  Kokkos::deep_copy(neighborhood_radii, h_radii);
}

std::string NeighborhoodQuality::info_string(const int tab_lev) const {
  std::ostringstream ss;
  const auto tab_str = indent_string(tab_lev);
  ss << tab_str << "NeighborhoodQuality:\n";
  ss << tab_str << "\tdegenerate targets = " << n_degenerate << " of "
     << n_targets << " (nonfinite " << n_nonfinite << ")\n";
  ss << tab_str << "\tworst target = " << worst_target
     << ", min lambda2/lambda1 = " << min_eig_ratio << "\n";
  ss << tab_str << "\tmin active neighbors = " << min_active
     << ", max r/h = " << max_r_over_h << "\n";
  return ss.str();
}

namespace {
/// eigenvalues (descending) of a symmetric 3x3 matrix, closed form
void sym3_eigenvalues(const Real a[3][3], Real ev[3]) {
  const Real p1 = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
  const Real q  = (a[0][0] + a[1][1] + a[2][2]) / 3;
  if (p1 == 0) {
    ev[0] = a[0][0];
    ev[1] = a[1][1];
    ev[2] = a[2][2];
  } else {
    const Real p2 = (a[0][0] - q) * (a[0][0] - q) +
                    (a[1][1] - q) * (a[1][1] - q) +
                    (a[2][2] - q) * (a[2][2] - q) + 2 * p1;
    const Real p  = std::sqrt(p2 / 6);
    Real b[3][3];
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        b[i][j] = (a[i][j] - (i == j ? q : 0)) / p;
    const Real detb = b[0][0] * (b[1][1] * b[2][2] - b[1][2] * b[2][1]) -
                      b[0][1] * (b[1][0] * b[2][2] - b[1][2] * b[2][0]) +
                      b[0][2] * (b[1][0] * b[2][1] - b[1][1] * b[2][0]);
    Real r = detb / 2;
    r      = std::max(Real(-1), std::min(Real(1), r));
    const Real phi = std::acos(r) / 3;
    ev[0]          = q + 2 * p * std::cos(phi);
    ev[2]          = q + 2 * p * std::cos(phi + 2 * M_PI / 3);
    ev[1]          = 3 * q - ev[0] - ev[2];
  }
  if (ev[0] < ev[1]) std::swap(ev[0], ev[1]);
  if (ev[1] < ev[2]) std::swap(ev[1], ev[2]);
  if (ev[0] < ev[1]) std::swap(ev[0], ev[1]);
}
}  // namespace

NeighborhoodQuality check_neighborhoods(
    const Kokkos::View<Real**>::HostMirror& src_crds,
    const Kokkos::View<Real**>::HostMirror& tgt_crds, const Neighborhoods& nn,
    const Params& params, const Real ratio_tol) {
  auto h_nl    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                   nn.neighbor_lists);
  auto h_radii = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), nn.neighborhood_radii);

  NeighborhoodQuality result;
  result.n_targets  = tgt_crds.extent(0);
  result.min_active = std::numeric_limits<Int>::max();
  const int pwr     = params.manifold_weight_pwr;
  for (Index i = 0; i < result.n_targets; ++i) {
    const Int nnbrs = h_nl(i, 0);
    const Real h    = h_radii(i);
    Real cov[3][3]  = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    Int active      = 0;
    bool finite     = std::isfinite(h);
    for (Int j = 0; j < nnbrs; ++j) {
      const Index s = h_nl(i, j + 1);
      Real d[3];
      Real r2 = 0;
      for (int k = 0; k < 3; ++k) {
        d[k] = src_crds(s, k) - tgt_crds(i, k);
        r2 += d[k] * d[k];
      }
      const Real r = std::sqrt(r2);
      finite       = finite and std::isfinite(r);
      result.max_r_over_h = std::max(result.max_r_over_h, r / h);
      const Real x = r / h;
      const Real w = (1 - x > 0) ? std::pow(1 - x, pwr) : 0;
      if (w > 0) ++active;
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) cov[a][b] += w * d[a] * d[b];
    }
    Real ratio = 0;
    if (finite) {
      Real ev[3];
      sym3_eigenvalues(cov, ev);
      ratio = (ev[0] > 0) ? ev[1] / ev[0] : 0;
    }
    result.min_active = std::min(result.min_active, active);
    const bool bad    = (not finite) or active < 3 or ratio < ratio_tol;
    if (not finite) ++result.n_nonfinite;
    if (bad) ++result.n_degenerate;
    if (ratio < result.min_eig_ratio or result.worst_target == LPM_NULL_IDX) {
      if (ratio <= result.min_eig_ratio) {
        result.min_eig_ratio = ratio;
        result.worst_target  = i;
      }
    }
  }
  return result;
}

std::string Neighborhoods::info_string(const int tab_lev) const {
  std::ostringstream ss;
  auto tab_str = indent_string(tab_lev);
  ss << tab_str << "gmls::Neighborhoods info:\n";
  tab_str += "\t";
  ss << tab_str << "neighbors.extents = (" << neighbor_lists.extent(0) << ", "
     << neighbor_lists.extent(1) << ")\n"
     << tab_str << "min_neighbors = " << n_min << "\n"
     << tab_str << "max_neighbors = " << n_max << "\n"
     << tab_str << "min_radius = " << r_min << "\n"
     << tab_str << "max_radius = " << r_max << "\n";
  return ss.str();
}

}  // namespace gmls
}  // namespace Lpm
