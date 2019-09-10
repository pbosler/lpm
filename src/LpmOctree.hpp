#ifndef LPM_OCTREE_HPP
#define LPM_OCTREE_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmCoords.hpp"
#include "Kokkos_Core.hpp"
#include <cmath>

namespace Lpm {

/** Data parallel octrees for surface reconstruction
    K. Zhou, et. al., IEE Trans. Vis. Comput. Graphics.
*/

typedef ko::View<Real[6]> box_type;

template <typename CPtView, typename Space> 
struct BoundingBox {
    typedef BoundingBox reducer;
    typedef Real value_type[6];
    typedef ko::View<Real[6],Space> result_view_type;
    
    KOKKOS_INLINE_FUNCTION
    void join(value_type& dest, const value_type& src) const {
        if (src[0] < dest[0]) dest[0] = src[0];
        if (src[1] > dest[1]) dest[1] = src[1];
        if (src[2] < dest[2]) dest[2] = src[2];
        if (src[3] > dest[3]) dest[3] = src[3];
        if (src[4] < dest[4]) dest[4] = src[4];
        if (src[5] > dest[5]) dest[5] = src[5];
    }
    
    KOKKOS_INLINE_FUNCTION
    void join(volatile value_type& dest, const volatile value_type& src) const {
        if (src[0] < dest[0]) dest[0] = src[0];
        if (src[1] > dest[1]) dest[1] = src[1];
        if (src[2] < dest[2]) dest[2] = src[2];
        if (src[3] > dest[3]) dest[3] = src[3];
        if (src[4] < dest[4]) dest[4] = src[4];
        if (src[5] > dest[5]) dest[5] = src[5];
    }
    
    KOKKOS_INLINE_FUNCTION
    void init(value_type& val) const {
        for (int i=0; i<6; ++i) {
            val[i] = (i%2==0 ? ko::reduction_identity<Real>::min() : ko::reduction_identity<Real>::max());
        }
    }
    
    KOKKOS_INLINE_FUNCTION
    value_type& reference() const {return *value.data();}
    
    KOKKOS_INLINE_FUNCTION
    result_view_type view() const {return value;}
    
    KOKKOS_INLINE_FUNCTION
    bool references_scalar() const {return references_scalar_v;}
    
    BoundingBox(value_type& val) : value(&val), references_scalar_v(true) {}
    
    BoundingBox(const result_view_type& val) : value(val), references_scalar_v(false) {}
    
    private:
        result_view_type value;
        bool references_scalar_v;
};

/**
    key = x1y1z1x2y2z2...xdydzd
    
    where bits xi, yi, zi, correspond to left (0) and right(1) of the midpoint of the octree
    node i's parent in the x,y,z direction
*/
template <typename CPtType> KOKKOS_INLINE_FUNCTION
int compute_key(const CPtType& pos, const int& level_depth) {
    int key = 0;
    /// assume root box is [-1,1]^3
    Real cx, cy, cz; // crds of box centroid
    cx = 0;
    cy = 0;
    cz = 0;
    for (int d=0; d<level_depth; ++d) {
        if (pos(0) > cx) key += ;
        if (pos(1) > cy) key += ;
        if (pos(2) > cz) key += ;
    } 
    return key;
}

struct OctreeLevel {
    
    ko::View<Index*[4]> vertex_nodes; /// vertex_nodes(i,:) = indices to nodes that share vertex i
    ko::View<Index*[2]> edge_verts; /// edge_verts(i,:) = [orig, dest] indices (into verts) for edge i
    ko::View<Index*[4]> face_edges; /// face_edges(i,:) = indices (into edges) bounding face i
    
    ko::View<int*> node_keys; /// 32-bit xyz-shuffle key
    ko::View<Index*[2]> node_inds; /** node_inds(i,0) is the first point index contained in node i
                                       node_inds(i,1) is the number of points contained in node i */
    ko::View<Index*> node_parents; /// node_parents(i) is the index (into nodes) of node i's parent
    ko::View<Index*[8]> node_kids; /// node_kids(i,:) are the indices of the children of node i
    ko::View<Index*[27]> node_neighbors; /** node_neighbors(i,:) are the indices of the nodes adjacent 
                                             to node i (including itself) */
    ko::View<Index*[8]> node_verts; /// node_verts(i,:) are indices (into vertices) of node i's vertices
    ko::View<Index*[12]> node_edges; /// node_edges(i,:) are the indices of node i's edges
    ko::View<Index*[6]> node_faces; /// node_faces(i,:) are the indices of the faces of node i

};



struct Octree {
    static constexpr int MAX_DEPTH = 10;
    
    OctreeLevel levels[MAX_DEPTH];
    
    box_type root_box;
};

}
#endif
