#include "LpmOctree.hpp"
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

namespace Lpm {
namespace Octree {

void Octree::init() {
    /// Build leaves
    NodeArrayD leaves(pts, max_depth);
    
    /// Build internal levels
    std::vector<NodeArrayInternal> internal_levels(max_depth-1);
    internal_levels(max_depth-1) = NodeArrayInternal(leaves);
    for (Int lev=max_depth-2; lev>=0; --lev) {
        internal_levels(lev) = NodeArrayInternal(internal_levels(lev+1));
    }
    
    /// Allocate full tree
    std::vector<Index*> nnodes_at_level(max_depth+1);
    nnodes_at_level(max_depth) = leaves.node_keys.extent(0);
    for (Int lev=max_depth-1; lev>=0; --lev) {
        nnodes_at_level(lev) = internal_levels(lev).node_keys.extent(0);
    }
    auto hbase = ko::create_mirror_view(base_address);
    Index sum = 0;
    for (Int lev=0; lev <= max_depth; ++lev) {
        hbase(lev) = sum;
        sum += nnodes_at_level(lev);
    }
    const Index nnodes_total = sum;
    node_keys = ko::View<key_type*>("node_keys", nnodes_total);
    node_pt_idx = ko::View<Index*>("node_pt_idx", nnodes_total);
    node_pt_ct = ko::View<Index*>("node_pt_ct", nnodes_total);
    node_parents = ko::View<Index*>("node_parents", nnodes_total);
    node_kids = ko::View<Index*[8]>("node_kids", nnodes_total);
    node_neighbors = ko::View<Index*[27]>("node_neighbors", nnodes_total);
    node_vertices = ko::View<Index*[8]>("node_vertices", nnodes_total);
    node_edges = ko::View<Index*[12]>("node_edges", nnodes_total);
    node_faces = ko::View<Index*[6]>("node_faces", nnodes_total);
    
    /// fill tree
    std::pair<size_t,size_t> view_range(hbase(max_depth), nnodes_total);
    auto key_view = ko::subview(node_keys, view_range);
    auto pt_idx_view = ko::subview(node_pt_idx, view_range);
    auto pt_ct_view = ko::subview(node_pt_ct, view_range);
    auto parent_view = ko::subview(node_parents, view_range);
    ko::deep_copy(key_view, leaves.node_keys);
    ko::deep_copy(pt_idx_view, leaves.node_pt_idx);
    ko::deep_copy(pt_ct_view, leaves.node_pt_ct);
    ko::deep_copy(parent_view, leaves.node_parent);
    
    for (Int lev=max_depth-1; lev>=0; --lev) {
        view_range = std::make_pair(hbase(lev), hbase(lev) + nnodes_at_level(lev));
        key_view = ko::subview(node_keys, view_range);
        pt_idx_view = ko::subview(node_pt_idx, view_range);
        pt_ct_view = ko::subview(node_pt_ct, view_range);
        auto kid_view = ko::subview(node_kids, view_range, ko::ALL());
        if (lev>0) {
            parent_view = ko::subview(node_parents, view_range);
            ko::deep_copy(parent_view, internal_levels(lev).node_parent);
        }        
        ko::deep_copy(key_view, internal_levels(lev).node_keys);
        ko::deep_copy(pt_idx_view, internal_levels(lev).node_pt_idx);
        ko::deep_copy(pt_ct_view, internal_levels(lev).node_pt_ct);
        ko::deep_copy(kid_view, internal_levels(lev).node_kids);
    }
    
    ko::deep_copy(box, leaves.box);
    ko::deep_copy(base_address, hbase);
}

}}
