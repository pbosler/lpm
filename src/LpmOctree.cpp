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
    ko::deep_copy(box, leaves.box);
    
    /// Build internal levels
    std::vector<NodeArrayInternal> internal_levels(max_depth-1);
    internal_levels[max_depth-1] = NodeArrayInternal(leaves);
    for (Int lev=max_depth-2; lev>0; --lev) {
        internal_levels[lev] = NodeArrayInternal(internal_levels[lev+1]);
    }
    
    /// Allocate full tree
    std::vector<Index> nnodes_at_level(max_depth+1);
    nnodes_at_level[max_depth] = leaves.node_keys.extent(0);
    for (Int lev=max_depth-1; lev>=0; --lev) {
        nnodes_at_level[lev] = internal_levels[lev].node_keys.extent(0);
    }
    auto hbase = ko::create_mirror_view(base_address);
    Index sum = 0;
    for (Int lev=0; lev <= max_depth; ++lev) {
        hbase(lev) = sum;
        sum += nnodes_at_level[lev];
    }
    ko::deep_copy(base_address, hbase);
    
    const Index nnodes_total = sum;
    node_keys = ko::View<key_type*>("node_keys", nnodes_total);
    node_pt_idx = ko::View<Index*>("node_pt_idx", nnodes_total);
    node_pt_ct = ko::View<Index*>("node_pt_ct", nnodes_total);
    node_parents = ko::View<Index*>("node_parents", nnodes_total);
    node_kids = ko::View<Index*[8]>("node_kids", nnodes_total);
    node_neighbors = ko::View<Index*[27]>("node_neighbors", nnodes_total);
    
    /// fill tree by concatenating levels
    auto view_range = std::make_pair(hbase(max_depth), nnodes_total);
    auto key_view = ko::subview(node_keys, view_range);
    auto pt_idx_view = ko::subview(node_pt_idx, view_range);
    auto pt_ct_view = ko::subview(node_pt_ct, view_range);
    auto parent_view = ko::subview(node_parents, view_range);
    ko::deep_copy(key_view, leaves.node_keys);
    ko::deep_copy(pt_idx_view, leaves.node_pt_idx);
    ko::deep_copy(pt_ct_view, leaves.node_pt_ct);
    ko::deep_copy(parent_view, leaves.node_parent);
    
    for (Int lev=max_depth-1; lev>=0; --lev) {
        view_range = std::make_pair(hbase(lev), hbase(lev) + nnodes_at_level[lev]);
        key_view = ko::subview(node_keys, view_range);
        pt_idx_view = ko::subview(node_pt_idx, view_range);
        pt_ct_view = ko::subview(node_pt_ct, view_range);
        auto kid_view = ko::subview(node_kids, view_range, ko::ALL());
        if (lev>0) {
            parent_view = ko::subview(node_parents, view_range);
            ko::deep_copy(parent_view, internal_levels[lev].node_parent);
        }        
        ko::deep_copy(key_view, internal_levels[lev].node_keys);
        ko::deep_copy(pt_idx_view, internal_levels[lev].node_pt_idx);
        ko::deep_copy(pt_ct_view, internal_levels[lev].node_pt_ct);
        ko::deep_copy(kid_view, internal_levels[lev].node_kids);
    }
    
    /// compute neighborhoods
    // setup root node
    auto root_neighbors = ko::subview(node_neighbors, 0, ko::ALL());
    auto hrn = ko::create_mirror_view(root_neighbors);
    for (int i = 0; i<13; ++i) {
        hrn(i) = NULL_IND;
    }
    hrn(13) = 0;
    for (int i=14; i<27; ++i) {
        hrn(i) = NULL_IND;
    }
    ko::deep_copy(root_neighbors, hrn);
    // compute neighbors
    for (int lev = 1; lev <= max_depth; ++lev) {
        auto neighborhood_policy = ExeSpaceUtils<>::get_default_team_policy(nnodes_at_level[lev],27);
        ko::parallel_for(neighborhood_policy, NeighborhoodFunctor(node_neighbors, node_keys, 
            node_kids, node_parents, base_address, lev, max_depth));
    }    
    
    if (do_connectivity) {
        node_vertices = ko::View<Index*[8]>("node_vertices", nnodes_total);
        node_edges = ko::View<Index*[12]>("node_edges", nnodes_total);
        node_faces = ko::View<Index*[6]>("node_faces", nnodes_total);  
        vertex_nodes = ko::View<Index*[8]>("vertex_nodes", 8*nnodes_total);
        edge_vertices = ko::View<Index*[2]>("edge_vertices",4*nnodes_total);
        
        /// compute vertex relations
        std::vector<ko::View<Index*[8]>> vert_nodes;
        std::vector<ko::View<Index*[8]>> node_verts;
        std::vector<Index> vbase(max_depth+1);
        std::vector<Index> nverts_at_level(max_depth+1);
        vbase[0] = 0;
        nverts_at_level[0] = 8;
        for (Int lev=1; lev<=max_depth; ++lev) {
            auto node_policy = ExeSpaceUtils<>::get_default_team_policy(nnodes_at_level[lev],8);
        
            // each node, in parallel, determines the owning node of its vertices
            ko::View<Index*[8]> vert_owners("vertex_owners", nnodes_at_level[lev]);
            ko::parallel_for(node_policy, VertexOwnerFunctor(vert_owners, node_keys, node_neighbors));
            // each node, in parallel, flags the vertices it owns and counts the number of vertices it owns
            ko::View<Int*[8]> vert_flags("vertex_flags", nnodes_at_level[lev]);
            ko::View<Int*> nverts_at_node("nverts_at_node", nnodes_at_level[lev]);
            ko::parallel_for(node_policy, VertexFlagFunctor(vert_flags, nverts_at_node, vert_owners));
            // compute the number of vertices at this level
            ko::parallel_reduce(nnodes_at_level[lev], KOKKOS_LAMBDA (const Index& t, Index& nv) {
                nv += nverts_at_node(t);
            }, nverts_at_level[lev]);
            for (Int i=0; i<lev; ++i) {
                vbase[lev] += nverts_at_level[i];
            }
            // determine the unique address of each vertex by its owning node
            ko::View<Index*> vert_address("vertex_address", nverts_at_level[lev]);
            ko::parallel_scan(nnodes_at_level[lev], 
                KOKKOS_LAMBDA (const Index& i, Int& ct, const bool& final_pass) {
                    const Index old_val = nverts_at_node(i);
                    if (final_pass) {
                        vert_address(i) = ct;
                    }
                    ct += old_val;
                });
            // allocate vertex arrays for this level
            vert_nodes[lev] = ko::View<Index*[8]>("vert_nodes_lev", nverts_at_level[lev]);
            node_verts[lev] = ko::View<Index*[8]>("node_verts_lev", nnodes_at_level[lev]);        
            // fill the level's vertex arrays
            ko::parallel_for(node_policy, VertexNodeFunctor(vert_nodes[lev], node_verts[lev], 
                vert_flags, vert_owners, vert_address, node_neighbors));
        }
        // concatenate levels into vertex_nodes, node_vertices
        auto root_vnodes = ko::subview(vertex_nodes, std::make_pair(0,8), ko::ALL);
        auto root_nverts = ko::subview(node_vertices, 0, ko::ALL);
        auto hroot_vn = ko::create_mirror_view(root_vnodes);
        auto hroot_nv = ko::create_mirror_view(root_nverts);
        for (Int i=0; i<8; ++i) {
            for (Int j=0; j<8; ++j) {
                hroot_vn(i,j) = (j != 7-i ? NULL_IND : 0);
            }
            hroot_nv(i) = i;
        }
        ko::deep_copy(root_vnodes, hroot_vn);
        ko::deep_copy(root_nverts, hroot_nv);
        for (Int lev=1; lev<=max_depth; ++lev) {
            auto vertex_view_range = std::make_pair(vbase[lev], vbase[lev] + nverts_at_level[lev]);
            auto node_view_range = std::make_pair(hbase(lev), hbase(lev) + nnodes_at_level[lev]);
            auto nv_view = ko::subview(node_vertices, node_view_range, ko::ALL());
            auto vn_view = ko::subview(vertex_nodes, vertex_view_range, ko::ALL());
            ko::deep_copy(nv_view, node_verts[lev]);
            ko::deep_copy(vn_view, vert_nodes[lev]);
        }
    
        // compute edge relations
        std::vector<ko::View<Index*[2]>> edge_verts;
        std::vector<ko::View<Index*[12]>> node_edges;
        std::vector<Index> ebase(max_depth+1);
        std::vector<Index> nedges_at_level(max_depth+1);
        ebase[0] = 0;
        nedges_at_level[0] = 12;
        for (Int lev=1; lev<=max_depth; ++lev) {
            auto node_policy = ko::TeamPolicy<>(nnodes_at_level[lev],12,4);
            ko::View<Index*[12]> edge_owners("edge_owners", nnodes_at_level[lev]);
            ko::parallel_for(node_policy, EdgeOwnerFunctor(edge_owners, node_keys, node_neighbors));
            ko::View<Int*[12]> edge_flags("edge_flags", nnodes_at_level[lev]);
            ko::View<Int*> nedges_at_node("nedges_at_node", nnodes_at_level[lev]);
            ko::parallel_for(node_policy, EdgeFlagFunctor(edge_flags, nedges_at_node, edge_owners));
            ko::parallel_reduce(nnodes_at_level[lev], KOKKOS_LAMBDA (const Index& t, Index& ne) {
                ne += nedges_at_node(t);
            }, nedges_at_level[lev]);
            ko::View<Index*> edge_address("edge_address", nedges_at_level[lev]);
            ko::parallel_scan(nnodes_at_level[lev], 
                KOKKOS_LAMBDA (const Index& t, Int& ct, const bool& final_pass) {
                    const Index old_val = nedges_at_node(t);
                    if (final_pass) {
                        edge_address(t) = ct;
                    }
                    ct += old_val;
                });
            for (Int i=0; i<lev; ++i) {
                ebase[lev] += nedges_at_level[i];
            }
            edge_verts[lev] = ko::View<Index*[2]>("edge_verts_lev", nedges_at_level[lev]);
            node_edges[lev] = ko::View<Index*[12]>("node_edges_lev", nnodes_at_level[lev]);
        }
    }
}

}}
