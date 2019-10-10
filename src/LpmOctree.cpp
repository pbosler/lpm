#include "LpmOctree.hpp"
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <bitset>
#include <cassert>

namespace Lpm {
namespace Octree {

void Tree::init() {
    assert(max_depth > 1 && max_depth <= MAX_OCTREE_DEPTH);
    
    /// Build leaves
    NodeArrayD leaves(pts, max_depth);
    ko::deep_copy(box, leaves.box);
    ko::deep_copy(pt_in_leaf, leaves.pt_in_node);
    ko::deep_copy(pt_orig_id, leaves.orig_ids);
    
    std::cout << "Octree leaves done.\n";
    std::cout << leaves.infoString();
    
    /// Build internal levels
    std::vector<NodeArrayInternal> internal_levels(max_depth-1);
    internal_levels[max_depth-1] = NodeArrayInternal(leaves);
    std::cout << "internal level " << max_depth - 1 << " done.\n";
    for (Int lev=max_depth-2; lev>0; --lev) {
        internal_levels[lev] = NodeArrayInternal(internal_levels[lev+1]);
        std::cout << "internal level " << lev << " done.\n";
    }
    std::cout << "Octree internal levels done.\n";
    
    /// Allocate full tree
    std::vector<Index> nnodes_at_level(max_depth+1);
    nnodes_at_level[max_depth] = leaves.node_keys.extent(0);
    for (Int lev=max_depth-1; lev>=0; --lev) {
        nnodes_at_level[lev] = internal_levels[lev].node_keys.extent(0);
    }
    auto nnh = ko::create_mirror_view(nnodes_per_level);
    auto hbase = ko::create_mirror_view(base_address);
    Index sum = 0;
    for (Int lev=0; lev <= max_depth; ++lev) {
        hbase(lev) = sum;
        nnh(lev) = nnodes_at_level[lev];
        sum += nnodes_at_level[lev];
    }
    ko::deep_copy(base_address, hbase);
    ko::deep_copy(nnodes_per_level, nnh);
    
    const Index nnodes_total = sum;
    node_keys = ko::View<key_type*>("node_keys", nnodes_total);
    node_pt_idx = ko::View<Index*>("node_pt_idx", nnodes_total);
    node_pt_ct = ko::View<Index*>("node_pt_ct", nnodes_total);
    node_parents = ko::View<Index*>("node_parents", nnodes_total);
    node_kids = ko::View<Index*[8]>("node_kids", nnodes_total);
    node_neighbors = ko::View<Index*[27]>("node_neighbors", nnodes_total);
    /// setup root node
    auto root_neighbors = ko::subview(node_neighbors, 0, ko::ALL());
    ko::parallel_for(27, KOKKOS_LAMBDA (const Int& i) { 
        node_neighbors(0,i) = (i != 13 ? NULL_IND : 0);
    });
    ko::parallel_for(1, KOKKOS_LAMBDA (const Int& i) {
        node_keys(0) = 0;
        node_pt_idx(0) = 0;
        node_pt_ct(0) = pts.extent(0);
        node_parents(0) = NULL_IND;
    });
    
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
    
    for (Int lev=max_depth-1; lev>0; --lev) {
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
        
        initVertices(nnodes_at_level, hbase);
        initEdges(nnodes_at_level, hbase);
        initFaces(nnodes_at_level, hbase);
    }
}

void Tree::initVertices(const std::vector<Index>& nnodes_at_level, const hbase_type& hbase) {
    /// compute vertex relations for each level
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
    /// concatenate levels into vertex_nodes, node_vertices
    Index nv = 0;
    for (Int lev=0; lev<=max_depth; ++lev) {
        nv += nverts_at_level[lev];
    }
    vertex_nodes = ko::View<Index*[8]>("vertex_nodes", nv);
    auto root_vnodes = ko::subview(vertex_nodes, std::make_pair(0,8), ko::ALL);
    auto root_nverts = ko::subview(node_vertices, 0, ko::ALL);
    ko::parallel_for(8, KOKKOS_LAMBDA (const Int& i) {
        for (Int j=0; j<8; ++j) {
            root_vnodes(i,j) = (j != 7-i ? NULL_IND : 0);
        }
        root_nverts(i) = i;
    });
    for (Int lev=1; lev<=max_depth; ++lev) {
        auto vertex_view_range = std::make_pair(vbase[lev], vbase[lev] + nverts_at_level[lev]);
        auto node_view_range = std::make_pair(hbase(lev), hbase(lev) + nnodes_at_level[lev]);
        auto nv_view = ko::subview(node_vertices, node_view_range, ko::ALL());
        auto vn_view = ko::subview(vertex_nodes, vertex_view_range, ko::ALL());
        ko::deep_copy(nv_view, node_verts[lev]);
        ko::deep_copy(vn_view, vert_nodes[lev]);
    }
}

void Tree::initEdges(const std::vector<Index>& nnodes_at_level, const hbase_type& hbase) {
    /// compute edge relations for each level
    std::vector<ko::View<Index*[2]>> everts(max_depth+1); // vector edge-vertex relations for each level
    std::vector<ko::View<Index*[12]>> nedges(max_depth+1); // vector of node-edge relations for each level
    std::vector<Index> ebase(max_depth+1); // starting address in full tree of each level's edges
    std::vector<Index> nedges_at_level(max_depth+1); // number of edges in each level
    ebase[0] = 0;
    nedges_at_level[0] = 12;
    for (Int lev=1; lev<=max_depth; ++lev) {
        auto node_policy = ko::TeamPolicy<>(nnodes_at_level[lev],12,4);
        // each node, in parallel, determines the owning node of its 12 edges
        ko::View<Index*[12]> edge_owners("edge_owners", nnodes_at_level[lev]);
        ko::parallel_for(node_policy, EdgeOwnerFunctor(edge_owners, node_keys, node_neighbors));
        // each node flags the edges it owns, and counts them
        ko::View<Int*[12]> edge_flags("edge_flags", nnodes_at_level[lev]);
        ko::View<Int*> nedges_at_node("nedges_at_node", nnodes_at_level[lev]);
        ko::parallel_for(node_policy, EdgeFlagFunctor(edge_flags, nedges_at_node, edge_owners));
        // compute the number of edges in this level
        ko::parallel_reduce(nnodes_at_level[lev], KOKKOS_LAMBDA (const Index& t, Index& ne) {
            ne += nedges_at_node(t);
        }, nedges_at_level[lev]);
        // compute the address of each node's first owned edge
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
        // construct edges at this level. edges are added by their owning node
        everts[lev] = ko::View<Index*[2]>("edge_verts_lev", nedges_at_level[lev]);
        nedges[lev] = ko::View<Index*[12]>("node_edges_lev", nnodes_at_level[lev]);
        ko::parallel_for(node_policy, EdgeNodeFunctor(everts[lev], nedges[lev], edge_flags,
            edge_owners, edge_address, node_neighbors, node_vertices));
    }
    /// concatenate levels to form node_edges and edge_vertices
    Index ne = 0;
    for (int lev = 0; lev<=max_depth; ++lev) {
        ne += nedges_at_level[lev];
    }
    edge_vertices = ko::View<Index*[2]>("edge_vertices", ne);
    auto evroot = ko::subview(edge_vertices, std::make_pair(0,12), ko::ALL());
    auto neroot = ko::subview(node_edges, 0, ko::ALL());
    auto evtable = ko::View<EdgeVerticesLUT>("EdgeVerticesLUT");
    ko::parallel_for(12, KOKKOS_LAMBDA (const Int& i) {
        for (Int j=0; j<2; ++j) evroot(i,j) = table_val(i,j, evtable);
        neroot(i) = i;
    });
    for (Int lev=1; lev<=max_depth; ++lev) {
        auto edge_view_range = std::make_pair(ebase[lev], ebase[lev] + nedges_at_level[lev]);
        auto node_view_range = std::make_pair(hbase(lev), hbase(lev) + nnodes_at_level[lev]);
        auto ev_view = ko::subview(edge_vertices, edge_view_range, ko::ALL());
        auto ne_view = ko::subview(node_edges, node_view_range, ko::ALL());
        ko::deep_copy(ev_view, everts[lev]);
        ko::deep_copy(ne_view, nedges[lev]);
    }
}

void Tree::initFaces(const std::vector<Index>& nnodes_at_level, const hbase_type& hbase) {
    /// compute face relations for each level
    std::vector<ko::View<Index*[4]>> fedges(max_depth+1); // vector of face-edge relations for each level
    std::vector<ko::View<Index*[6]>> nfaces(max_depth+1); // vector of node-face relations for each level
    std::vector<Index> fbase(max_depth+1); // starting address in full tree of each level's faces
    std::vector<Index> nfaces_at_level(max_depth+1); // number of faces in each level
    fbase[0] = 0;
    nfaces_at_level[0] = 6;
    for (Int lev=1; lev<=max_depth; ++lev) {
        auto node_policy = ko::TeamPolicy<>(nnodes_at_level[lev],6);
        // each node determines the node that owns its 6 faces
        ko::View<Index*[6]> face_owners("face_owners", nnodes_at_level[lev]);
        ko::parallel_for(node_policy, FaceOwnerFunctor(face_owners, node_keys, node_neighbors));
        // each node flags the faces it owns, and counts them
        ko::View<Int*[6]> face_flags("face_flags", nnodes_at_level[lev]);
        ko::View<Int*> nfaces_at_node("nfaces_at_node", nnodes_at_level[lev]);
        ko::parallel_for(node_policy, FaceFlagFunctor(face_flags, nfaces_at_node, face_owners));
        // compute the number of faces in this level
        ko::parallel_reduce(nnodes_at_level[lev], KOKKOS_LAMBDA (const Index& t, Index& nf) {
            nf += nfaces_at_node(t);
        }, nfaces_at_level[lev]);
        // scan to compute the address of each node's first owned face
        ko::View<Index*> face_address("face_address", nfaces_at_level[lev]);
        ko::parallel_scan(nnodes_at_level[lev],
            KOKKOS_LAMBDA (const Index& t, Int& ct, const bool& final_pass) {
                const Index old_val = nfaces_at_node(t);
                if (final_pass) {
                    face_address(t) = ct;
                }
                ct += old_val;
            });
        for (Int i=0; i<lev; ++i) {
            fbase[lev] += nfaces_at_level[i];
        }
        // construct faces at this level. 
        fedges[lev] = ko::View<Index*[4]>("face_edges_lev", nfaces_at_level[lev]);
        nfaces[lev] = ko::View<Index*[6]>("node_faces_lev", nnodes_at_level[lev]);
        ko::parallel_for(node_policy, FaceNodeFunctor(fedges[lev], nfaces[lev], face_flags, 
            face_owners, face_address, node_neighbors, node_edges));
    }
    /// concatenate levels to form face_edges and node_faces
    Index nf = 0;
    for (Int lev=0; lev<=max_depth; ++lev) {
        nf += nfaces_at_level[lev];
    }
    face_edges = ko::View<Index*[4]>("face_edges", nf);
    auto feroot = ko::subview(face_edges, std::make_pair(0,6), ko::ALL);
    auto nfroot = ko::subview(node_faces, 0, ko::ALL());
    auto fetable = ko::View<FaceEdgesLUT>("FaceEdgesLUT");
    ko::parallel_for(6, KOKKOS_LAMBDA (const Int& i) {
        for (Int j=0; j<4; ++j) feroot(i,j) = table_val(i,j, fetable);
        nfroot(i) = i;
    });
    for (Int lev=1; lev <= max_depth; ++lev) {
        auto face_view_range = std::make_pair(fbase[lev], fbase[lev] + nfaces_at_level[lev]);
        auto node_view_range = std::make_pair(hbase(lev), hbase(lev) + nnodes_at_level[lev]);
        auto fe_view = ko::subview(face_edges, face_view_range, ko::ALL());
        auto nf_view = ko::subview(node_faces, node_view_range, ko::ALL());
        ko::deep_copy(fe_view, fedges[lev]);
        ko::deep_copy(nf_view, nfaces[lev]);
    }
}

std::string Tree::infoString() const {
    auto root_box_host = ko::create_mirror_view(box);
    auto node_keys_host = ko::create_mirror_view(node_keys);
    auto node_pt_host = ko::create_mirror_view(node_pt_idx);
    auto node_ct_host = ko::create_mirror_view(node_pt_ct);
    auto node_parent_host = ko::create_mirror_view(node_parents);
    auto node_kids_host = ko::create_mirror_view(node_kids);
    auto node_neighbors_host = ko::create_mirror_view(node_neighbors);
    auto pt_leaf_host = ko::create_mirror_view(pt_in_leaf);
    auto pt_orig_host = ko::create_mirror_view(pt_orig_id);
    auto ba_host = ko::create_mirror_view(base_address); 
    auto nnodes_host = ko::create_mirror_view(nnodes_per_level);
    
    ko::deep_copy(root_box_host, box);
    ko::deep_copy(node_keys_host, node_keys);
    ko::deep_copy(node_pt_host, node_pt_idx);
    ko::deep_copy(node_ct_host, node_pt_ct);
    ko::deep_copy(node_parent_host, node_parents);
    ko::deep_copy(node_kids_host, node_kids);
    ko::deep_copy(node_neighbors_host, node_neighbors);
    ko::deep_copy(pt_leaf_host, pt_in_leaf);
    ko::deep_copy(pt_orig_host, pt_orig_id);
    ko::deep_copy(ba_host, base_address);
    ko::deep_copy(nnodes_host, nnodes_per_level);
    
    if (do_connectivity) {
        auto node_verts_host = ko::create_mirror_view(node_vertices);
        auto node_edges_host = ko::create_mirror_view(node_edges);
        auto node_faces_host = ko::create_mirror_view(node_faces);
        auto vertnodes_host = ko::create_mirror_view(vertex_nodes);
        auto edgeverts_host = ko::create_mirror_view(edge_vertices);
        auto faceedges_host = ko::create_mirror_view(face_edges);
        ko::deep_copy(node_verts_host, node_vertices);
        ko::deep_copy(node_edges_host, node_edges);
        ko::deep_copy(node_faces_host, node_faces);
        ko::deep_copy(vertnodes_host, vertex_nodes);
        ko::deep_copy(edgeverts_host, edge_vertices);
        ko::deep_copy(faceedges_host, face_edges);
    }
    
    std::ostringstream ss;
    ss << "Octree info:\n";
    for (Int lev=0; lev<=max_depth; ++lev) {
        ss << "\tlevel " << lev << "\n";
        for (Index i=0; i<nnodes_host[lev]; ++i) {
            const std::string tabstr("\t\t");
            ss << tabstr << "node(" << i << "): key = " << std::bitset<3*MAX_OCTREE_DEPTH>(node_keys_host(i))
               << " pt_start = " << node_pt_host(i) << " pt_ct = " << node_ct_host(i)
               << " parent = " << node_parent_host(i)
               << " kids = ";
            for (int j=0; j<8; ++j) {
                ss << node_kids_host(i,j) << " ";
            }
            ss << " box = ";
            const BBox nbox = box_from_key(node_keys_host(i), root_box_host(), lev, max_depth);
            ss << nbox;
        }
    }
    return ss.str();
}

}}
