#include "LpmNodeArrayInternal.hpp"
#include "LpmOctreeKernels.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <bitset>
#include <exception>
#include <cassert>

namespace Lpm {
namespace Octree {

void NodeArrayInternal::initFromLeaves(NodeArrayD& leaves) {
    const Index nparents = leaves.node_keys.extent(0)/8;
    assert(leaves.node_keys.extent(0)%8 == 0);    
    
    ko::View<key_type*> pkeys("parent_keys", nparents);
    ko::View<Index*[2]> pinds("point_inds", nparents);
    ko::parallel_for(nparents, ParentNodeFunctor(pkeys, pinds, leaves.node_keys, leaves.node_pt_inds,
        level, max_depth));
// DEBUG SECTION
    auto pkeys_host = ko::create_mirror_view(pkeys);
    auto pinds_host = ko::create_mirror_view(pinds);
    ko::deep_copy(pkeys_host, pkeys);
    ko::deep_copy(pinds_host, pinds);
    Index npoints = 0;
    ko::parallel_reduce(nparents, KOKKOS_LAMBDA (const Index& i, Index& ct) {
        ct += pinds(i,1);
    }, npoints);
    assert(npoints == leaves.sorted_pts.extent(0));
    
//     for (Index i=0; i<nparents; ++i) {
//         std::cout << "pkeys(" << i << ") = " << pkeys_host(i) << " pt_start = " << pinds_host(i,0)
//             << " pt_count = " << pinds_host(i,1) << " parent key = " << parent_key(pkeys_host(i), level, max_depth) << "\n";
//     }
// END DEBUG SECTION        
    
    ko::View<Index*> nsiblings("nsiblings", nparents);
    ko::parallel_for(ko::RangePolicy<NodeSiblingCounter::MarkTag>(0,nparents), 
        NodeSiblingCounter(nsiblings, pkeys, level, max_depth));
// DEBUG SECTION
//     auto nsibs_host = ko::create_mirror_view(nsiblings);
//     ko::deep_copy(nsibs_host, nsiblings);
//     std::cout << "nsiblings (before scan) = (";
//     for (Index i=0; i<nparents; ++i) {
//         std::cout << nsibs_host(i) << (i<nparents-1 ? " " : ")\n");
//     }
// END DEBUG SECTION
    ko::parallel_scan(ko::RangePolicy<NodeSiblingCounter::ScanTag>(0,nparents),
        NodeSiblingCounter(nsiblings, pkeys, level, max_depth));
// DEBUG SECTION
//     ko::deep_copy(nsibs_host, nsiblings);
//     std::cout << "nsiblings (after scan) = (";
//     for (Index i=0; i<nparents; ++i) {
//         std::cout << nsibs_host(i) << (i<nparents-1 ? " " : ")\n");
//     }
// END DEBUG SECTION

    
    n_view_type nnodes_view = ko::subview(nsiblings, nparents-1);
    auto nnhost = ko::create_mirror_view(nnodes_view);
    ko::deep_copy(nnhost, nnodes_view);
    const Index nnodes = nnhost();

// DEBUG SECTION
//     std::cout << "from " << nparents << " parents at lower level, need " << nnodes << " siblings at this level.\n";
// END DEBUG SECTION        
    
    node_keys = ko::View<key_type*>("node_keys", nnodes);
    node_pt_inds = ko::View<Index*[2]>("node_pt_inds",nnodes);
    node_parents = ko::View<Index*>("node_parents", nnodes);
    node_kids = ko::View<Index*[8]>("node_kids", nnodes);
    
    ko::parallel_for(nparents, NodeArrayInternalFunctor(node_keys, node_pt_inds, node_parents, 
        node_kids, leaves.node_parents, nsiblings, pkeys, pinds, leaves.node_keys, 
        level, max_depth));
// DEBUG SECTION
    npoints = 0;
    const auto loc_pinds = node_pt_inds;
    ko::parallel_reduce(nnodes, KOKKOS_LAMBDA (const Index& i, Index& ct) {
        ct += loc_pinds(i,1);
    }, npoints);
    assert(npoints == leaves.sorted_pts.extent(0));
// END DEBUG SECTION    
}

void NodeArrayInternal::initFromLower(NodeArrayInternal& lower) {
    const Index nparents = lower.node_keys.extent(0)/8;
    assert(lower.node_keys.extent(0)%8 == 0);
    
    ko::View<key_type*> pkeys("parent_keys", nparents);
    ko::View<Index*[2]> pinds("point_inds", nparents);
    ko::parallel_for(nparents, ParentNodeFunctor(pkeys, pinds, lower.node_keys,
        lower.node_pt_inds, level, max_depth));
    
    ko::View<Index*> nsiblings("nsiblings", nparents);
    ko::parallel_for(ko::RangePolicy<NodeSiblingCounter::MarkTag>(0,nparents),
        NodeSiblingCounter(nsiblings, pkeys, level, max_depth));
    ko::parallel_scan(ko::RangePolicy<NodeSiblingCounter::ScanTag>(0,nparents),
        NodeSiblingCounter(nsiblings, pkeys, level, max_depth));
    
    n_view_type nnodes_view = ko::subview(nsiblings, nparents-1);
    auto nnhost = ko::create_mirror_view(nnodes_view);
    ko::deep_copy(nnhost, nnodes_view);
    const Index nnodes = nnhost();
    
    node_keys = ko::View<key_type*>("node_keys", nnodes);
    node_pt_inds = ko::View<Index*[2]>("node_pt_inds", nnodes);
    node_parents = ko::View<Index*>("node_parents", nnodes);
    node_kids = ko::View<Index*[8]>("node_kids", nnodes);
    
    ko::parallel_for(nparents, NodeArrayInternalFunctor(node_keys, node_pt_inds,
        node_parents, node_kids, lower.node_parents, nsiblings, pkeys, pinds,
        lower.node_keys, level, max_depth));
}


std::string NodeArrayInternal::infoString(const bool& verbose) const {
    std::ostringstream ss;
    ss << "NodeArrayInternal (level " << level << " of " << max_depth << ") info:\n";
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
        for (Index i=0; i<node_keys.extent(0); ++i) {
            ss << "\t\tkey = " << keys(i) << " pts_start = " << pt_inds(i,0) << " pts_ct = " << pt_inds(i,1) 
               << " parent = " << parents(i) << " kids = (";
            for (int j=0; j<8; ++j) {
                ss << kids(i,j) << (j<7 ? " " : ")\n");
            }
        }
    }
    return ss.str();
}

}}
