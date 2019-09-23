#ifndef LPM_OCTREE_HPP
#define LPM_OCTREE_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmCoords.hpp"
#include "LpmBox3d.hpp"
#include "LpmOctreeUtil.hpp"
#include "LpmKokkosUtil.hpp"
#include "Kokkos_Core.hpp"
#include <cmath>

namespace Lpm {
namespace Octree {
/** 
    K. Zhou, et. al., 2011. Data-parallel octrees for surface reconstruction, 
    IEEE Trans. Vis. Comput. Graphics 17(5): 669--681. DOI: 10.1109/TVCG.2010.75 
*/


/**
    Node array at depth D
    Input: pts = kokkos view of 3d xyz coordinates
    
    Step 1: compute the bounding box
    
            BBox box;
            ko::parallel_reduce(pts.extent(0), BoxFunctor(pts), BBoxReducer<Space>(box));
        
    Step 2: compute the key of each point and encode the pt id with its key

            ko::parallel_for(pts.extent(0), KOKKOS_LAMBDA (const Index& i) {
                auto pos = ko::subview(pts, i, ko::ALL());
                const key_type key = compute_key(pos, D);
                codes(i) = encode(key, i);
            });
    
    Step 3: Sort by key, rearrange points into sorted order.
            a. sort the encoded key-id pairs by key
            
            ko::sort(codes);
            ko::View<Real*[3]> temp_pts("pts", pts.extent(0));
            
            b. rearrange pts array into sorted key order

            ko::parallel_for(pts.extent(0), PermuteKernel(temp_pts, pts, codes));
            pts = temp_pts;
    
    Step 4: Consolidate unique nodes.
            a. Mark encoded keys as either unique (1) or duplicate (0).
    
            ko::parallel_for(ko::RangePolicy<MarkDuplicates::MarkTag>(0, pts.extent(0)), MarkDuplicates(flags, codes));
                    
            b. Find indices of the unique nodes and total number of unique nodes
                
            ko::parallel_scan(ko::RangePolicy<MarkDuplicates::ScanTag>(0, pts.extent(0)), MarkDuplicates(flags, codes));
                
            c. Make unique nodes and track their particle indices
            
            ko::parallel_for(pts.extent(0), NodeSetupKernel(unique_keys, pt_inds, flags, codes));
            
    Step 5: Build NodeArrayD (1 of 2)
            a. Count nodes that need to be added.
            b. Calculate addresses of nodes to be added.
    
    Step 6: Build NodeArrayD (2 of 2)
        
        
    
*/
class NodeArrayD {
    public:
    ko::View<Real*[3]> pts; /// point coordinates in R3 (input).
    Int level; /// depth of this node array in the octree (input).
    Int max_depth; /// maximum depth of octree.
    
    ko::View<key_type*> node_keys; /// node_keys(i) = shuffled xyz key of node i
    ko::View<Index*> node_pt_idx; /// node_pt_idx(i) = index of first point contained in node i
    ko::View<Index*> node_pt_ct; /// node_pt_ct(i) = number of points contained in node i
    ko::View<key_type*[8]> node_kids; /// 
    ko::View<key_type*> node_parent; ///
    ko::View<Index*> pt_in_node; /// pt_in_node(i) = index of the node that contains point i
        
    NodeArrayD(ko::View<Real*[3]>& p, const Int& d, const Int& md=MAX_OCTREE_DEPTH) : pts(p), level(d), max_depth(md), 
        pt_in_node("pt_in_node", p.extent(0)) {init();}
    
    NodeArrayD(NodeArrayD& lower_level) : pts(lower_level.pts), level(lower_level.level-1),
         max_depth(lower_level.max_depth), node_kids("node_kids", lower_level.node_keys.extent(0)/8) { 
        initFromLower(lower_level); }
    
    protected:
    
    /** 
    */
    void initFromLower(NodeArrayD& ll) {
        ko::View<code_type*> parent_keys("parent_keys", ll.node_keys.extent(0));
        ko::parallel_for(ll.node_keys.extent(0), KOKKOS_LAMBDA (const Index& i) {
            parent_keys(i) = encode(parent_key(ll.node_keys(i), level, max_depth), 0);
        });
        ko::View<Index*> node_flags("node_flags", ll.node_keys.extent(0));
        ko::parallel_for(ko::RangePolicy<MarkDuplicates::MarkTag>(0, ll.node_keys.extent(0)),
            MarkDuplicates(node_flags, parent_keys));
        ko::parallel_scan(ko::RangePolicy<MarkDuplicates::ScanTag>(0, ll.node_keys.extent(0)),
            MarkDuplicates(node_flags, parent_keys));
        n_view_type node_count = ko::subview(node_flags, ll.node_keys.extent(0)-1);
        auto node_count_host = ko::create_mirror_view(node_count);
        ko::deep_copy(node_count_host, node_count);
            
    }

    /**
        Listing 1:  Initializer for lowest level of octree
    */
    void init() {
        /// step 1
        ko::View<BBox> box("bbox"); /// Bounding box
        ko::parallel_reduce(pts.extent(0), BoxFunctor(pts), BBoxReducer<>(box()));
        
        /// step 2
        ko::View<code_type*> pt_codes("pt_codes",pts.extent(0));
        ko::parallel_for(pts.extent(0), KOKKOS_LAMBDA (const Index& i) {
            auto pos = ko::subview(pts, i, ko::ALL());
            const key_type key = compute_key(pos, level, MAX_OCTREE_DEPTH, box());
            pt_codes(i) = encode(key, i);
        });
        ko::sort(codes);
        
        /// step 3
        ko::View<Real*[3]> sort_pts("sorted_pts", pts.exgtent(0));
        ko::parallel_for(pts.extent(0), PermuteKernel(sort_pts, pts, codes));
        pts = sort_pts;
        
        /// step 4
        ko::View<Index*> node_flags("node_flags",pts.extent(0));
        ko::parallel_for(ko::RangePolicy<MarkDuplicates::MarkTag>(0,pts.extent(0)), 
            MarkDuplicates(node_flags, codes));
        ko::parallel_scan(ko::RangePolicy<MarkDuplicates::ScanTag>(0,pts.extent(0)),
            MarkDuplicates(node_flags, codes));
        n_view_type node_count = ko::subview(node_flags, pts.extent(0)-1);
        auto node_count_host = ko::create_mirror_view(node_count);
        ko::deep_copy(node_count_host, node_count);
        
        ko::View<key_type*> ukeys("keys", node_count_host());
        ko::View<Index*[2]> pt_inds("pt_inds", node_count_host());
        ko::parallel_for(pts.extent(0), UniqueNodeKernel(ukeys, pt_inds, node_flags, codes));
        
        /// step 5
        ko::View<Index*> node_nums("node_nums", node_count_host());
        ko::View<Index*> node_address("node_address", node_count_host());
        ko::parallel_for(ko::RangePolicy<NodeAddressKernel::MarkTag>(0, node_count_host()),
            NodeAddressKernel(node_nums, node_address, ukeys, level, MAX_OCTREE_DEPTH));
        ko::parallel_scan(ko::RangePolicy<NodeAddressKernel::ScanTag>(0, node_count_host()),
            NodeAddressKernel(node_nums, node_address, ukeys, level, MAX_OCTREE_DEPTH));
        
        /// step 6
        node_count = ko::subview(node_address, node_count_host()-1);
        ko::deep_copy(node_count_host, node_count);
        node_keys = ko::View<key_type*>("keys", node_count_host());
        node_pt_idx = ko::View<Index*>("pt_start_index", node_count_host());
        node_pt_ct = ko::View<Index*>("pt_count", node_count_host());
        node_parent = ko::View<Index*>("node_parent", node_count_host());
        ko::parallel_for(ukeys.extent(0), NodeArrayKernel(node_keys, node_pt_idx, node_pt_ct, pt_in_node, ukeys, 
            node_address, pt_inds, level, max_depth));
    };    
};


}}
#endif
