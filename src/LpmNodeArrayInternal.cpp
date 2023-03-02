#include "LpmNodeArrayInternal.hpp"

#include <bitset>
#include <cassert>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "LpmOctreeKernels.hpp"

namespace Lpm {
namespace Octree {

void NodeArrayInternal::initFromLeaves(NodeArrayD& leaves) {
  Index nnodes;
  if (level == 0) {
    nnodes = 1;
    // build root node
    node_keys = ko::View<key_type*>("node_keys", 1);
    node_pt_inds = ko::View<Index* [2]>("node_pt_inds", 1);
    node_parents = ko::View<Index*>("node_parents", 1);
    node_kids = ko::View<Index* [8]>("node_kids", 1);

    auto node_keys_host = ko::create_mirror_view(node_keys);
    node_keys_host(0) = 0;
    ko::deep_copy(node_keys, node_keys_host);

    auto node_pt_inds_host = ko::create_mirror_view(node_pt_inds);
    auto leaves_pt_inds_host = ko::create_mirror_view(leaves.node_pt_inds);
    ko::deep_copy(leaves_pt_inds_host, leaves.node_pt_inds);
    node_pt_inds_host(0, 0) = 0;
    node_pt_inds_host(0, 1) = 0;
    for (Int i = 0; i < 8; ++i) {
      node_pt_inds_host(0, 1) += leaves_pt_inds_host(i, 1);
    }
    ko::deep_copy(node_pt_inds, node_pt_inds_host);

    auto node_parents_host = ko::create_mirror_view(node_parents);
    node_parents_host(0) = NULL_IND;
    ko::deep_copy(node_parents, node_parents_host);

    auto node_kids_host = ko::create_mirror_view(node_kids);
    for (int i = 0; i < 8; ++i) {
      node_kids_host(0, i) = i;
    }
    ko::deep_copy(node_kids, node_kids_host);

    auto leaves_parents = leaves.node_parents;
    ko::parallel_for(
        8, KOKKOS_LAMBDA(const Index& i) { leaves_parents(i) = 0; });
  } else {
    const Index nparents = leaves.node_keys.extent(0) / 8;
    assert(leaves.node_keys.extent(0) % 8 == 0);

    ko::View<key_type*> pkeys("parent_keys", nparents);
    ko::View<Index* [2]> pinds("point_inds", nparents);
    ko::parallel_for(nparents,
                     ParentNodeFunctor(pkeys, pinds, leaves.node_keys,
                                       leaves.node_pt_inds, level, max_depth));

    //     #ifdef LPM_ENABLE_DEBUG
    //         Index npoints = 0;
    //         ko::parallel_reduce(nparents, KOKKOS_LAMBDA (const Index& i,
    //         Index& ct) {
    //             ct += pinds(i,1);
    //         }, npoints);
    //         assert(npoints == leaves.sorted_pts.extent(0));
    //     #endif

    ko::View<Index*> nsiblings("nsiblings", nparents);
    ko::parallel_for(ko::RangePolicy<NodeSiblingCounter::MarkTag>(0, nparents),
                     NodeSiblingCounter(nsiblings, pkeys, level, max_depth));
    ko::parallel_scan(ko::RangePolicy<NodeSiblingCounter::ScanTag>(0, nparents),
                      NodeSiblingCounter(nsiblings, pkeys, level, max_depth));

    n_view_type nnodes_view = ko::subview(nsiblings, nparents - 1);
    auto nnhost = ko::create_mirror_view(nnodes_view);
    ko::deep_copy(nnhost, nnodes_view);
    nnodes = nnhost();

    node_keys = ko::View<key_type*>("node_keys", nnodes);
    node_pt_inds = ko::View<Index* [2]>("node_pt_inds", nnodes);
    node_parents = ko::View<Index*>("node_parents", nnodes);
    node_kids = ko::View<Index* [8]>("node_kids", nnodes);

    ko::parallel_for(nparents,
                     NodeArrayInternalFunctor(
                         node_keys, node_pt_inds, node_parents, node_kids,
                         leaves.node_parents, nsiblings, pkeys, pinds,
                         leaves.node_keys, level, max_depth));
  }
#ifdef LPM_ENABLE_DEBUG
  std::cout << "NodeArrayInternal::initFromLeaves: nnodes = " << nnodes
            << " out of " << pintpow8(level) << " possible.\n";
  Index npoints = 0;
  const auto loc_pinds = node_pt_inds;
  ko::parallel_reduce(
      nnodes,
      KOKKOS_LAMBDA(const Index& i, Index& ct) { ct += loc_pinds(i, 1); },
      npoints);
  assert(npoints == leaves.sorted_pts.extent(0));
  std::cout << "NodeArrayInternal::initFromLeaves: npoints check = "
            << (npoints == leaves.sorted_pts.extent(0) ? " pass\n" : "FAIL\n");
#endif
}

void NodeArrayInternal::initFromLower(NodeArrayInternal& lower) {
  Index nnodes;
  if (level == 0) {
    nnodes = 1;
    // build root node
    node_keys = ko::View<key_type*>("node_keys", 1);
    node_pt_inds = ko::View<Index* [2]>("node_pt_inds", 1);
    node_parents = ko::View<Index*>("node_parents", 1);
    node_kids = ko::View<Index* [8]>("node_kids", 1);

    auto node_keys_host = ko::create_mirror_view(node_keys);
    node_keys_host(0) = 0;
    ko::deep_copy(node_keys, node_keys_host);

    auto node_pt_inds_host = ko::create_mirror_view(node_pt_inds);
    auto lower_pt_inds_host = ko::create_mirror_view(lower.node_pt_inds);
    ko::deep_copy(lower_pt_inds_host, lower.node_pt_inds);
    node_pt_inds_host(0, 0) = 0;
    node_pt_inds_host(0, 1) = 0;
    for (Int i = 0; i < 8; ++i) {
      node_pt_inds_host(0, 1) += lower_pt_inds_host(i, 1);
    }
    ko::deep_copy(node_pt_inds, node_pt_inds_host);

    auto node_parents_host = ko::create_mirror_view(node_parents);
    node_parents_host(0) = NULL_IND;
    ko::deep_copy(node_parents, node_parents_host);

    auto node_kids_host = ko::create_mirror_view(node_kids);
    for (int i = 0; i < 8; ++i) {
      node_kids_host(0, i) = i;
    }
    ko::deep_copy(node_kids, node_kids_host);

    auto lower_parents = lower.node_parents;
    ko::parallel_for(
        8, KOKKOS_LAMBDA(const Index& i) { lower_parents(i) = 0; });
  } else {
    const Index nparents = lower.node_keys.extent(0) / 8;
    assert(lower.node_keys.extent(0) % 8 == 0);

    ko::View<key_type*> pkeys("parent_keys", nparents);
    ko::View<Index* [2]> pinds("point_inds", nparents);
    ko::parallel_for(nparents,
                     ParentNodeFunctor(pkeys, pinds, lower.node_keys,
                                       lower.node_pt_inds, level, max_depth));

    ko::View<Index*> nsiblings("nsiblings", nparents);
    ko::parallel_for(ko::RangePolicy<NodeSiblingCounter::MarkTag>(0, nparents),
                     NodeSiblingCounter(nsiblings, pkeys, level, max_depth));
    ko::parallel_scan(ko::RangePolicy<NodeSiblingCounter::ScanTag>(0, nparents),
                      NodeSiblingCounter(nsiblings, pkeys, level, max_depth));

    n_view_type nnodes_view = ko::subview(nsiblings, nparents - 1);
    auto nnhost = ko::create_mirror_view(nnodes_view);
    ko::deep_copy(nnhost, nnodes_view);
    nnodes = nnhost();

    node_keys = ko::View<key_type*>("node_keys", nnodes);
    node_pt_inds = ko::View<Index* [2]>("node_pt_inds", nnodes);
    node_parents = ko::View<Index*>("node_parents", nnodes);
    node_kids = ko::View<Index* [8]>("node_kids", nnodes);

    ko::parallel_for(nparents,
                     NodeArrayInternalFunctor(
                         node_keys, node_pt_inds, node_parents, node_kids,
                         lower.node_parents, nsiblings, pkeys, pinds,
                         lower.node_keys, level, max_depth));
  }
#ifdef LPM_ENABLE_DEBUG
  std::cout << "NodeArrayInternal::initFromLower: nnodes = " << nnodes
            << " out of " << pintpow8(level) << " possible.\n";
  Index npts_lower = 0;
  Index npts_check = 0;
  const auto loc_lower_inds = lower.node_pt_inds;
  ko::parallel_reduce(
      lower.node_pt_inds.extent(0),
      KOKKOS_LAMBDA(const Index& i, Index& ct) { ct += loc_lower_inds(i, 1); },
      npts_lower);
  const auto loc_inds = node_pt_inds;
  ko::parallel_reduce(
      nnodes,
      KOKKOS_LAMBDA(const Index& i, Index& ct) { ct += loc_inds(i, 1); },
      npts_check);
  assert(npts_lower == npts_check);
  std::cout << "NodeArrayInternal::initFromLower: npoints check = "
            << (npts_check == npts_lower ? " pass\n" : "FAIL\n");
#endif
}

std::string NodeArrayInternal::infoString(const bool& verbose) const {
  std::ostringstream ss;
  ss << "NodeArrayInternal (level " << level << " of " << max_depth
     << ") info:\n";
  ss << "\tnnodes = " << node_keys.extent(0) << "\n";
  if (verbose) {
    auto keys = ko::create_mirror_view(node_keys);
    auto pt_inds = ko::create_mirror_view(node_pt_inds);
    auto parents = ko::create_mirror_view(node_parents);
    auto kids = ko::create_mirror_view(node_kids);
    ko::deep_copy(keys, node_keys);
    ko::deep_copy(pt_inds, node_pt_inds);
    ko::deep_copy(parents, node_parents);
    ko::deep_copy(kids, node_kids);

    ss << "\tNodes:\n";
    for (Index i = 0; i < node_keys.extent(0); ++i) {
      ss << "\t\tkey = " << std::bitset<32>(keys(i))
         << " pts_start = " << pt_inds(i, 0) << " pts_ct = " << pt_inds(i, 1)
         << " parent = " << parents(i) << " kids = (";
      for (int j = 0; j < 8; ++j) {
        ss << kids(i, j) << (j < 7 ? " " : ")\n");
      }
    }
  }
  return ss.str();
}

}  // namespace Octree
}  // namespace Lpm
