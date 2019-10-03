#ifndef LPM_NODE_ARRAY_INTERNAL_HPP
#define LPM_NODE_ARRAY_INTERNAL_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmKokkosUtil.hpp"
#include "LpmBox3d.hpp"
#include "LpmOctreeUtil.hpp"
#include "LpmNodeArrayD.hpp"
#include "Kokkos_Core.hpp"
#include <iostream>
#include <sstream>

namespace Lpm {
namespace Octree {

struct ParentNodeFunctor {
    // output
    ko::View<key_type*> keys_out;
    // input
    ko::View<key_type*> keys_from_lower;
    Int level;
    Int lower_level;
    Int max_depth;
    
    ParentNodeFunctor(ko::View<key_type*>& ko, const ko::View<key_type*>& kl, const Int& lev, const Int& md) :
        keys_out(ko), keys_from_lower(kl), level(lev), lower_level(lev+1), max_depth(md) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const Index& i) const {
        const key_type& my_first_kid = 8*i;
        const key_type& my_key = parent_key(keys_from_lower(my_first_kid), lower_level, max_depth);
        keys_out(i) = my_key;
    }
};

struct NodeFillInternal {
    // output
    ko::View<Index*> pt_idx; // start point indices for parent keys
    ko::View<Index*> pt_ct; // point counts for parent keys
    ko::View<Index*[8]> kids_out; // address of kids in next lower level
    ko::View<Index*> parents_from_lower; // address of parents in this level, viewed from lower level
    // input
    ko::View<key_type*> keys_in; // only non-empty keys
    ko::View<key_type*> keys_from_lower; // keys from lower level
    ko::View<Index*> pt_start_from_lower;
    ko::View<Index*> pt_count_from_lower;
    ko::View<Index*> node_address;
    Int level;
    Int max_depth;
    
    NodeFillInternal(ko::View<Index*>& ptsi, ko::View<Index*>& ptsc, 
        ko::View<Index*[8]>& kkids, ko::View<Index*>& plow, const ko::View<key_type*>& pkeys,
        const ko::View<key_type*>& klow, const ko::View<Index*>& ptsi_low, 
        const ko::View<Index*>& ptsc_low, const ko::View<Index*>& na, const Int& lev, const Int& md) :
        pt_idx(ptsi), pt_ct(ptsc), kids_out(kkids), parents_from_lower(plow),
        keys_in(pkeys), keys_from_lower(klow), pt_start_from_lower(ptsi_low), pt_count_from_lower(ptsc_low),
        node_address(na), level(lev), max_depth(md) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const member_type& mbr) const {
        const Index i = mbr.league_rank();
        const Index address = node_address(i) + local_key(keys_in(i), level, max_depth);
        const Index first_kid = binarySearchKeys(keys_in(i), keys_from_lower);
        pt_idx(address) = pt_start_from_lower(first_kid);
        ko::parallel_for(ko::TeamThreadRange(mbr, 8), KOKKOS_LAMBDA (const Index& j) {
            parents_from_lower(first_kid+j) = address;
            kids_out(address,j) = first_kid + j;
        });
        ko::parallel_reduce(ko::TeamThreadRange(mbr,8), KOKKOS_LAMBDA (const Index& j, Index& c) {
            c += pt_count_from_lower(first_kid+j);
        }, pt_ct(address));
    }
};

class NodeArrayInternal {
    public:
        Int level;
        Int max_depth;
        
        ko::View<key_type*> node_keys; // keys of nodes at this level
        ko::View<Index*> node_pt_idx; // index of first point contained by nodes
        ko::View<Index*> node_pt_ct; // number of points contained by each node at this level
        
        ko::View<Index*> node_parent; // address of parents of nodes at this level (into level-1 NodeArrayInternal)
        ko::View<Index*[8]> node_kids; // address of children (in level+1 NodeArray)
        
        NodeArrayInternal() {}
        
        NodeArrayInternal(NodeArrayD& leaves) : level(leaves.level-1), 
            max_depth(leaves.max_depth) { initFromLeaves(leaves); }
        
        NodeArrayInternal(NodeArrayInternal& lower) : level(lower.level-1),
            max_depth(lower.max_depth) { initFromLower(lower); }
    
        std::string infoString() const;
    
        void initFromLeaves(NodeArrayD& leaves) {
            /// get keys for parents of nodes included at lower level
            const Index nparents = leaves.node_keys.extent(0)/8;
            ko::View<key_type*> pkeys("parent_keys",nparents);
            ko::parallel_for(nparents, ParentNodeFunctor(pkeys, leaves.node_keys, level, max_depth));
            
            
            /// augment keys to make sure included nodes' siblings are also included
            ko::View<Index*> node_nums("node_nums", nparents);
            ko::View<Index*> node_address("node_address", nparents);
            ko::parallel_for(ko::RangePolicy<NodeAddressFunctor::MarkTag>(0,nparents),
                NodeAddressFunctor(node_nums, node_address, pkeys, level, max_depth));
            ko::parallel_scan(ko::RangePolicy<NodeAddressFunctor::ScanTag>(0,nparents),
                NodeAddressFunctor(node_nums, node_address, pkeys, level, max_depth));
            
            n_view_type last_address = ko::subview(node_address, nparents-1);
            auto la_host = ko::create_mirror_view(last_address);
            ko::deep_copy(la_host, last_address);
            const key_type nnodes = la_host() + 8;
                        
            std::ostringstream ss;
            ss << "node_keys" << level;
            node_keys = ko::View<key_type*>(ss.str(), nnodes);
            ss.str("");
            ss << "node_pt_idx" << level;
            node_pt_idx = ko::View<Index*>(ss.str(), nnodes);
            ss.str("");
            ss << "node_pt_ct" << level;
            node_pt_ct = ko::View<Index*>(ss.str(), nnodes);
            ss.str(""); 
            if (level > 1) {
                ss << "node_parent" << level;
                node_parent = ko::View<Index*>(ss.str(), nnodes);
                ss.str("");
            }
            ss << "node_kids" << level;
            node_kids = ko::View<Index*[8]>(ss.str(), nnodes);
            
            auto setup_policy = ExeSpaceUtils<>::get_default_team_policy(nparents, 8);
            ko::parallel_for(setup_policy, NodeSetupFunctor(node_keys, node_address, pkeys, level, max_depth));
                        
            auto fill_policy = ExeSpaceUtils<>::get_default_team_policy(nparents, 8);
            ko::parallel_for(fill_policy, NodeFillInternal(node_pt_idx, node_pt_ct, node_kids,
                leaves.node_parent, pkeys, leaves.node_keys, leaves.node_pt_idx, leaves.node_pt_ct, node_address,
                level, max_depth));
        }
        
        void initFromLower(NodeArrayInternal& lower) {
            /// get keys for parents of nodes included at lower level
            const Index nparents = lower.node_keys.extent(0)/8;
            ko::View<key_type*> pkeys("parent_keys", nparents);
            ko::parallel_for(nparents, ParentNodeFunctor(pkeys, lower.node_keys, level, max_depth));
            
            key_type nnodes;
            ko::View<Index*> node_nums("node_nums", nparents);
            ko::View<Index*> node_address("node_address", nparents);
            if (level > 0) {
                /// augment keys to make sure included nodes' siblings are also included   
                ko::parallel_for(ko::RangePolicy<NodeAddressFunctor::MarkTag>(0,nparents),
                    NodeAddressFunctor(node_nums, node_address, pkeys, level, max_depth));
                ko::parallel_scan(ko::RangePolicy<NodeAddressFunctor::ScanTag>(0,nparents),
                    NodeAddressFunctor(node_nums, node_address, pkeys, level, max_depth));
                n_view_type last_address = ko::subview(node_address, nparents-1);
                auto la_host = ko::create_mirror_view(last_address);
                ko::deep_copy(la_host, last_address);
                nnodes = la_host() + 8;
            }
            else {
                nnodes = 1;
            }
            
            
            std::ostringstream ss;
            ss << "node_keys" << level;
            node_keys = ko::View<key_type*>(ss.str(),nnodes);
            ss.str("");
            
            ss << "node_pt_idx" << level;
            node_pt_idx = ko::View<Index*>(ss.str(), nnodes);
            ss.str("");
            
            ss << "node_pt_ct" << level;
            node_pt_ct = ko::View<Index*>(ss.str(), nnodes);
            ss.str("");
            if (level>0) {
                ss << "node_parent" << level;
                node_parent = ko::View<Index*>(ss.str(), nnodes);
                ss.str("");
            }
            ss << "node_kids" << level;
            node_kids = ko::View<Index*[8]>(ss.str(), nnodes);
            
            auto setup_policy = ExeSpaceUtils<>::get_default_team_policy(nparents,8);
            ko::parallel_for(setup_policy, NodeSetupFunctor(node_keys, node_address, pkeys, level, max_depth));
            
            
            ko::parallel_for(setup_policy, NodeFillInternal(node_pt_idx, node_pt_ct, node_kids, 
                lower.node_parent, pkeys, lower.node_keys, lower.node_pt_idx, lower.node_pt_ct, node_address, 
                level, max_depth));
        }
    protected:
};

}}
#endif