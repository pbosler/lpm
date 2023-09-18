#include "lpm_compadre.hpp"

#include <iostream>
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
  h_radii = Kokkos::create_mirror_view(neighborhood_radii);
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
#ifndef NDEBUG
  std::cout << info_string();
#endif
}

void Neighborhoods::update_neighbors(const host_crd_view host_src_crds,
                const host_crd_view host_tgt_crds,
                const Params& params,
                const bool verbose) {
  auto point_cloud_search = Compadre::PointCloudSearch(host_src_crds);
  const bool dry_run = false;
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
