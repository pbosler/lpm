#include "LpmNodeArrayD.hpp"
#include "LpmOctreeUtil.hpp"
#include "LpmOctreeKernels.hpp"
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
    
#ifdef LPM_ENABLE_DEBUG
    std::cout << "NodeArrayD::init step 1 of 6 done.\n";
#endif
    
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
    
#ifdef LPM_ENABLE_DEBUG
    std::cout << "NodeArrayD::init step 2 of 6 done.\n";
#endif    
    /// step 3: Sort point array
    /**
        Input: points, sorted point codes
        Process: rearrange points according to sorted codes.
            Kernel: PermuteFunctor
                Loop over: codes, points
        Output: sorted point array, with original id array to allow unsort later
    */
    ko::sort(pt_codes);
    
    ko::parallel_for(npts, PermuteFunctor(sorted_pts, orig_ids, presorted_pts, pt_codes));
    
#ifdef LPM_ENABLE_DEBUG
    std::cout << "NodeArrayD::init step 3 of 6 done.\n";
#endif
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
    
#ifdef LPM_ENABLE_DEBUG
    std::cout << "NodeArrayD::init step 4 of 6 done.\n";
#endif
    
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
#ifdef LPM_ENABLE_DEBUG
    std::cout << "NodeArrayD::init step 5 of 6 done.\n";
#endif   
    /// step 6: Build NodeArrayD
//     std::cout << "allocated " << nnodes << " leaf nodes.\n";
    node_keys = ko::View<key_type*>("node_keys", nnodes);
    node_pt_inds = ko::View<Index*[2]>("node_pt_inds", nnodes);
    node_parents = ko::View<Index*>("node_parents", nnodes);
    ko::parallel_for(unode_count, NodeArrayDFunctor(node_keys, node_pt_inds, node_parents, 
        pt_in_node, nsiblings, ukeys, uinds, depth));
    
#ifdef LPM_ENABLE_DEBUG
    std::cout << "NodeArrayD::init step 6 of 6 done.\n";
#endif    
};

std::string NodeArrayD::infoString(const bool& verbose) const {
    std::ostringstream ss;
    ss << "NodeArrayD info:\n";
    ss << "\tdepth = " << depth << "\n";
    ss << "\tnpts = " << sorted_pts.extent(0) << "\n";
    ss << "\tnnodes = " << node_keys.extent(0) << "\n";
    auto rbox = ko::create_mirror_view(box);
    ko::deep_copy(rbox, box);
    ss << "\tbbox = " << rbox();
    if (verbose) {
        auto keys = ko::create_mirror_view(node_keys);
        auto pt_inds = ko::create_mirror_view(node_pt_inds);
        auto parents = ko::create_mirror_view(node_parents);
        auto pt_node = ko::create_mirror_view(pt_in_node);
        auto old_ids = ko::create_mirror_view(orig_ids);
        
        ko::deep_copy(keys, node_keys);
        ko::deep_copy(pt_inds, node_pt_inds);
        ko::deep_copy(parents, node_parents);
        ko::deep_copy(pt_node, pt_in_node);
        ko::deep_copy(old_ids, orig_ids);
        
        for (Index i=0; i<node_keys.extent(0); ++i) {
            ss << "\t\t" << i << ": key = " << keys(i) << " pt_start = " << pt_inds(i,0) << " pt_ct = " << pt_inds(i,1)
               << " parent key = " << parent_key(keys(i), depth, depth) << " parent index = " << parents(i) << "\n";
        }
//         ss << "\tpoints:\n";
//         for (Index i=0; i<pt_in_node.extent(0); ++i) {
//             ss << "\t\t" << i << ": in node " << pt_node(i) << " old_id = " << old_ids(i) << "\n";
//         }
    }
    return ss.str();
}

}}
