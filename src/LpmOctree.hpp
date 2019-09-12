#ifndef LPM_OCTREE_HPP
#define LPM_OCTREE_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmCoords.hpp"
#include "LpmOctreeUtil.hpp"
#include "Kokkos_Core.hpp"
#include <cmath>

namespace Lpm {
namespace Octree {
/** Data parallel octrees for surface reconstruction
    K. Zhou, et. al., IEE Trans. Vis. Comput. Graphics.
*/






struct OctreeLevel {
    typedef ko::View<uint_fast64_t*> code_view;
    typedef ko::View<Index*[2]> pt_index_view;
    typedef typename index_view_type parent_view;
    typedef ko::View<Index*[8]> tree_view;
    typedef ko::View<Index*[27]> neighbors_view;

    struct CodeTag {};
    struct SortTag {};

    Int depth;
    Index nnodes;
    Index npts;
    box_type b;
    
    code_view codes; /// 32-bit xyz-shuffle key + 32-bit indices of points
    pt_index_view inds; /** inds(i,0) is the first point index contained in node i
                                       inds(i,1) is the number of points contained in node i */
    parent_view parents; /// parents(i) is the index (into nodes) of node i's parent
    tree_view kids; /// kids(i,:) are the indices of the children of node i
//     neighbors_view neighbors; /** neighbors(i,:) are the indices of the nodes adjacent 
//                                              to node i (including itself) */
    points_view points;                                            

//     ko::View<Index*[8]> verts; /// verts(i,:) are indices (into vertices) of node i's vertices
//     ko::View<Index*[12]> edges; /// edges(i,:) are the indices of node i's edges
//     ko::View<Index*[6]> faces; /// faces(i,:) are the indices of the faces of node i


    OctreeLevel(const Int& d, const box_type& bb, points_view pts, code_view cv, pt_index_view iv, 
        parent_view pv, tree_view tv) : depth(d), nnodes(pintpow2(3*d)), npts(pts.extent(0)),
        points(pts), codes(cv), inds(iv), parents(pv), kids(tv), neighbors(nbv), b(bb) {}
    
    
    KOKKOS_INLINE_FUNCTION
    void operator(const CodeTag&, const uint_fast32_t& i) const {
        auto pos = ko::subview(points, i, ko::ALL());
        const uint_fast32_t key = compute_key(pos, depth);
        codes(i) = ((key << 32) + i); 
    }
  
    KOKKOS_INLINE_FUNCTION
    void operator(const SortTag&, const member_type& mbr) const {
        
    }
    
};



struct Octree {
    static constexpr int MAX_DEPTH = 10;
    
    OctreeLevel levels[MAX_DEPTH];
    
    ko::View<Index*[4]> vertex_nodes; /// vertex_nodes(i,:) = indices to nodes that share vertex i
    ko::View<Index*[2]> edge_verts; /// edge_verts(i,:) = [orig, dest] indices (into verts) for edge i
    ko::View<Index*[4]> face_edges; /// face_edges(i,:) = indices (into edges) bounding face i
    
    box_type root_box;
    Uint max_levels;
};

}}
#endif
