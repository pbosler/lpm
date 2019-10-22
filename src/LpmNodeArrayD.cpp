#include "LpmNodeArrayD.hpp"
#include <string>
#include <iostream>
#include <iomanip>
#include <bitset>
#include <sstream>
#include "Kokkos_Sort.hpp"

namespace Lpm {
namespace Octree {

void NodeArrayD::init(const ko::View<Real*[3]>& presorted_pts)  {
    assert(depth > 0 && depth <= MAX_OCTREE_DEPTH);
    const Index npts = presorted_pts.extent(0);
    /// step 1: Determine bounding box of point set
    /**
        Input: points
        Process: 
            Kernel: BoxFunctor
                Reduction
                Loop over: Points
        Output: bounding box
    */
    ko::parallel_reduce(npts, BoxFunctor(presorted_pts), BBoxReducer<Dev>(box));
    
    /// step 2: Encode node key/point index pairs
    /**
        Input: points
        Process: compute 32-bit node key of each point, concatenate 32-bit point id into 64-bit code
        Kernel : EncodeFunctor
        Loop over: points
        Output: encoded point key/id pairs
        Sort code array
    */
    ko::View<code_type*> pt_codes("pt_codes",npts);
    ko::parallel_for(npts, EncodeFunctor(pt_codes, presorted_pts, box, depth));
    ko::sort(pt_codes);

    /// step 3: Sort point array
    /**
        Input: points, sorted point codes
        Process: rearrange points according to sorted codes.
            Kernel: PermuteFunctor
                Loop over: codes, points
        Output: sorted point array, with original id array to allow unsort later
    */
    ko::parallel_for(npts, PermuteFunctor(sorted_pts, orig_ids, presorted_pts, pt_codes));

    /// step 4: Determine points contained by each node
    /**
        Input: point codes
        Process: 
            Kernel: MarkDuplicates
                Loop over: point codes                
                a. Flag each unique node using encoded keys (flag = 1 for unique node; 0 otherwise)
                b. Inclusive scan to count the number of unique nodes 
                c. After scan, flags(npts-1) = count of nodes that have points
                
            Kernel: UniqueNodeFunctor
                Loop over: point codes
                For each node, find index of first contained point, number of points contained
        Output: 
            unique node keys 
            point start, count
    */
    ko::View<Index*> node_flags("node_flags",npts);
    ko::parallel_for(ko::RangePolicy<MarkDuplicates::MarkTag>(0,npts), 
        MarkDuplicates(node_flags, pt_codes));
    ko::parallel_scan(ko::RangePolicy<MarkDuplicates::ScanTag>(0,npts),
        MarkDuplicates(node_flags, pt_codes));
        
    n_view_type unode_count_view = ko::subview(node_flags, npts-1);
    auto un_count_host = ko::create_mirror_view(unode_count_view);
    ko::deep_copy(un_count_host, unode_count_view);
    const Index unode_count = un_count_host();        
    
    ko::View<key_type*> ukeys("ukeys", unode_count);
    ko::View<Index*[2]> uinds("uinds", unode_count);
    ko::parallel_for(npts, UniqueNodeFunctor(ukeys, uinds, node_flags, pt_codes));
    
    /// step 5: Ensure that each node has a full set of siblings
    /**
        Reserve space for each unique parent to have a full set of 8 children.
        Equivalently: Reserve space for each unique node to have a full set of 8 siblings.
        Process:
            Kernel: NodeSiblingCounter
                Loop over: unique keys
                a. Flag each node with a new parent (8 = new parent, 0 otherwise)
                b. Inclusive scan to count the number of nodes that need to be allocated
    */
    ko::View<Index*> nsiblings("nsiblings", unode_count);
    ko::parallel_for(ko::RangePolicy<NodeSiblingCounter::MarkTag>(0, unode_count), 
        NodeSiblingCounter(nsiblings, ukeys, depth, depth));
    ko::parallel_scan(ko::RangePolicy<NodeSiblingCounter::ScanTag>(0, unode_count),
        NodeSiblingCounter(nsiblings, ukeys, depth, depth));
    
    n_view_type nnodes_view = ko::subview(nsiblings, unode_count-1);
    auto nnhost = ko::create_mirror_view(nnodes_view);
    ko::deep_copy(nnhost, nnodes_view);
    const Index nnodes = nnhost();
    
    /// step 6: Build NodeArrayD
    node_keys = ko::View<key_type*>("node_keys", nnodes);
    node_pt_inds = ko::View<Index*[2]>("node_pt_inds", nnodes);
    node_parents = ko::View<Index*>("node_parents", nnodes);
    auto node_policy = ExeSpaceUtils<>::get_default_team_policy(unode_count,8);
    ko::parallel_for(node_policy, NodeArrayDFunctor(node_keys, node_pt_inds, nsiblings, ukeys, uinds, depth));
};

std::string NodeArrayD::infoString() const {
    std::ostringstream ss;
    auto bv = ko::create_mirror_view(box);
    ko::deep_copy(bv, box);
    ss << "NodeArrayD info:\n";
    ss << "\tbounding box: " << bv() << "\tedge_len <= " << longestEdge(bv()) << " ar = " << boxAspectRatio(bv()) << "\n";
    ss << "\tdepth = " << depth << "\n";
    
    auto keys = ko::create_mirror_view(node_keys);
    auto pinds = ko::create_mirror_view(node_pt_inds);
    auto parents = ko::create_mirror_view(node_parents);
    ko::deep_copy(keys, node_keys);
    ko::deep_copy(pinds, node_pt_inds);
    ko::deep_copy(parents, node_parents);
    const Index nnodes = node_keys.extent(0);
    ss << "\tNodes: " << nnodes << "\n";
    for (Index i=0; i<nnodes; ++i) {
        const BBox nbox = box_from_key(keys(i), bv(), depth, depth);
        ss << "node(" << std::setw(8)<< i << "): key = " << std::setw(8) << keys(i) 
           << " " << std::bitset<32>(keys(i));
        ss << " pt_start = " << pinds(i,0) << " pt_ct = " << pinds(i,1)
           << " parent = " << parents(i) 
           << "\tbox = " << nbox << "\tedge_len <= " << longestEdge(nbox) << " ar = " << boxAspectRatio(nbox) << "\n";
    }
    
//     auto pin = ko::create_mirror_view(pt_in_node);
//     auto oid = ko::create_mirror_view(orig_ids);
//     ko::deep_copy(pin, pt_in_node);
//     ko::deep_copy(oid, orig_ids);
//     const Index npts = sorted_pts.extent(0);
//     for (Index i=0; i<npts; ++i) {
//         ss << "point(" << i << ") is in node " << pin(i) << " orig_id = " << oid(i) << "\n";
//     }
    
    return ss.str();
}

}}
