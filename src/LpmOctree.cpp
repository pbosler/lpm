#include "LpmOctree.hpp"

#include <bitset>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

namespace Lpm {
namespace Octree {

/**
    Each level will be built in serial, from bottom to top.


    NodeArrayD = leaves, level = max_depth
    std::vector<NodeArrayInternal> internal_levels contains levels for lev=1 to
   lev=max_depth-1 root is built explicitly
*/
void Tree::initNodes() {
  assert(max_depth >= 1 && max_depth <= MAX_OCTREE_DEPTH);

  /// Build leaves
  NodeArrayD leaves(presorted_pts, max_depth);
  sorted_pts = leaves.sorted_pts;
  pt_in_leaf = leaves.pt_in_node;
  pt_orig_id = leaves.orig_ids;
  const Index nleaves = leaves.node_keys.extent(0);

#ifdef LPM_ENABLE_DEBUG
  std::cout << "Octree leaves done.\n";
  std::cout << leaves.infoString();
#endif

  /// Build internal levels (including root)
  std::vector<NodeArrayInternal> internal_levels(max_depth);
  internal_levels[max_depth - 1] = NodeArrayInternal(leaves);
  for (int lev = max_depth - 2; lev >= 0; --lev) {
    internal_levels[lev] = NodeArrayInternal(internal_levels[lev + 1]);
  }

  /// allocate full tree arrays
  nnodes_per_level_host = ko::create_mirror_view(nnodes_per_level);
  nnodes_total = nleaves;
  nnodes_per_level_host(max_depth) = nleaves;
  for (int lev = 0; lev < max_depth; ++lev) {
    nnodes_per_level_host(lev) = internal_levels[lev].node_keys.extent(0);
    nnodes_total += nnodes_per_level_host(lev);
  }
  base_address_host = ko::create_mirror_view(base_address);
  base_address_host(0) = 0;
  for (int lev = 1; lev <= max_depth; ++lev) {
    base_address_host(lev) =
        base_address_host(lev - 1) + nnodes_per_level_host(lev - 1);
  }
  ko::deep_copy(nnodes_per_level, nnodes_per_level_host);
  ko::deep_copy(base_address, base_address_host);

  node_keys = ko::View<key_type*>("node_keys", nnodes_total);
  node_pt_inds = ko::View<Index* [2]>("node_pt_inds", nnodes_total);
  node_parents = ko::View<Index*>("node_parents", nnodes_total);
  node_kids = ko::View<Index* [8]>("node_kids", nnodes_total);
  node_neighbors = ko::View<Index* [27]>("node_neighbors", nnodes_total);

  /// concatenate levels
  for (int lev = 0; lev < max_depth; ++lev) {
    auto dest_range =
        std::make_pair(base_address_host(lev),
                       base_address_host(lev) + nnodes_per_level_host(lev));
    ko::deep_copy(ko::subview(node_keys, dest_range),
                  internal_levels[lev].node_keys);
    ko::deep_copy(ko::subview(node_pt_inds, dest_range, ko::ALL()),
                  internal_levels[lev].node_pt_inds);
    ko::deep_copy(ko::subview(node_parents, dest_range),
                  internal_levels[lev].node_parents);
    ko::deep_copy(ko::subview(node_kids, dest_range, ko::ALL()),
                  internal_levels[lev].node_kids);
  }
  auto dest_range = std::make_pair(
      base_address_host(max_depth),
      base_address_host(max_depth) + nnodes_per_level_host(max_depth));
  ko::deep_copy(ko::subview(node_keys, dest_range), leaves.node_keys);
  ko::deep_copy(ko::subview(node_pt_inds, dest_range, ko::ALL()),
                leaves.node_pt_inds);
  ko::deep_copy(ko::subview(node_parents, dest_range), leaves.node_parents);
  auto leaf_kids = ko::subview(node_kids, dest_range, ko::ALL());
  ko::parallel_for(
      nleaves, KOKKOS_LAMBDA(const Index& i) {
        for (int j = 0; j < 8; ++j)
          leaf_kids(i, j) = NULL_IND;  // all leaves have no children
      });
  auto root_neighbors = ko::subview(node_neighbors, 0, ko::ALL());
  auto root_neighbors_host = ko::create_mirror_view(root_neighbors);
  for (int j = 0; j < 27; ++j) {
    root_neighbors_host(j) = (j == 13 ? 0 : NULL_IND);
  }
  ko::deep_copy(root_neighbors, root_neighbors_host);
  for (int lev = 1; lev <= max_depth; ++lev) {
    const ko::RangePolicy<> range_pol(
        base_address_host(lev),
        base_address_host(lev) + nnodes_per_level_host(lev));
    ko::parallel_for(range_pol,
                     NeighborhoodFunctor(node_neighbors, node_keys, node_kids,
                                         node_parents, lev, max_depth));
  }

#ifdef LPM_ENABLE_DEBUG
  for (int lev = 0; lev <= max_depth; ++lev) {
    Index null_neighbor_ct = 0;
    const ko::RangePolicy<> range_pol(
        base_address_host(lev),
        base_address_host(lev) + nnodes_per_level_host(lev));
    auto neighbors = node_neighbors;
    ko::parallel_reduce(
        range_pol,
        KOKKOS_LAMBDA(const Index& i, Index& ct) {
          for (int j = 0; j < 27; ++j) {
            if (neighbors(i, j) == NULL_IND) ct += 1;
          }
        },
        null_neighbor_ct);
    std::cout << "null neighbors at level " << lev << ": " << null_neighbor_ct
              << "\n";
  }
#endif

  if (do_connectivity) {
    initVertices();
  }
}

void Tree::initVertices() {
  /// vertex_owner(t,i) = address of node that owns node t's ith vertex
  ko::View<Index* [8]> vertex_owners("vertex_owners", nnodes_total);
  /// nverts_at_node(t) = count of vertices owned by node t
  ko::View<Index*> nverts_at_node("nverts_at_node", nnodes_total);
  ko::View<Int* [8]> vertex_flags("vertex_flags", nnodes_total);
  /// vertex_address(t) = address in vertex array of first vertex owned by node
  /// t
  ko::View<Index*> vertex_address("vertex_address", nnodes_total);

  Index nverts_total = 8;
  for (int lev = 1; lev <= max_depth; ++lev) {
    const Index lev_slice_start = base_address_host(lev);
    const Index lev_slice_end =
        base_address_host(lev) + nnodes_per_level_host(lev);

    ko::parallel_for(
        ko::RangePolicy<VertexSetupFunctor::OwnerTag>(lev_slice_start,
                                                      lev_slice_end),
        VertexSetupFunctor(vertex_owners, vertex_flags, nverts_at_node,
                           vertex_address, node_keys, node_neighbors,
                           lev_slice_start));

    Index nverts_this_level = 0;
    ko::parallel_reduce(
        ko::RangePolicy<VertexSetupFunctor::ReduceTag>(lev_slice_start,
                                                       lev_slice_end),
        VertexSetupFunctor(vertex_owners, vertex_flags, nverts_at_node,
                           vertex_address, node_keys, node_neighbors,
                           lev_slice_start),
        nverts_this_level);

    nverts_total += nverts_this_level;

    ko::parallel_scan(
        ko::RangePolicy<VertexSetupFunctor::ScanTag>(lev_slice_start,
                                                     lev_slice_end),
        VertexSetupFunctor(vertex_owners, vertex_flags, nverts_at_node,
                           vertex_address, node_keys, node_neighbors,
                           lev_slice_start));
  }
}

}  // namespace Octree
}  // namespace Lpm
