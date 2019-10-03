#ifndef LPM_OCTREE_HPP
#define LPM_OCTREE_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmKokkosUtil.hpp"
#include "LpmUtilities.hpp"
#include "LpmBox3d.hpp"
#include "LpmOctreeUtil.hpp"
#include "LpmOctreeLUT.hpp"
#include "LpmNodeArrayD.hpp"
#include "LpmNodeArrayInternal.hpp"
#include "Kokkos_Core.hpp"
#include <cassert>

namespace Lpm {
namespace Octree {



class Octree {
    public:
        ko::View<Real*[3]> pts;
        Int max_depth;
        
        ko::View<BBox> box;
        
        ko::View<key_type*> node_keys;
        ko::View<Index*> node_pt_idx;
        ko::View<Index*> node_pt_ct;
        ko::View<Index*> node_parents;
        ko::View<Index*[8]> node_kids;
        ko::View<Index*[27]> node_neighbors;
        ko::View<Index*[8]> node_vertices;
        ko::View<Index*[12]> node_edges;
        ko::View<Index*[6]> node_faces;
        
        ko::View<Index*> pt_in_leaf;
        ko::View<Index*[8]> vertex_nodes;
        ko::View<Index*[2]> edge_vertices;
        ko::View<Index*[4]> face_edges;
        ko::View<Index*> base_address;
        
        ko::View<Index[8][27]> parent_table;
        ko::View<Index[8][27]> child_table;
        
        Octree(const ko::View<Real*[3]>& p, const Int& md) : pts(p), max_depth(md), 
            pt_in_leaf("pt_in_leaf", p.extent(0)), base_address("base_address", max_depth+1),
            box("bbox"), parent_table("parent_table"), child_table("child_table") {
                init();
            }
        
    protected:
        void init();
    
};

struct NeighborhoodFunctor {
    // output
    ko::View<Index*[27]> neighbors;
    // input
    ko::View<key_type*> keys;
    ko::View<Index*[8]> kids;
    ko::View<Index*> parents;
    ko::View<Index*> base_address;
    Int level; // assume > 0
    Int max_depth;
    
    NeighborhoodFunctor(ko::View<Index*[27]>& n, const ko::View<key_type*>& k, const ko::View<Index*[8]>& c,
        const ko::View<Index*>& p, const ko::View<Index*>& b, const Int& l, const Int& m) :
        neighbors(n), keys(k), kids(c), parents(p), base_address(b), level(l), max_depth(m) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const member_type& mbr) const {
        const Index t = base_address(level) + mbr.league_rank();
        const Index p = parents(t);
        const key_type i = local_key(keys(t), level, max_depth);
        ko::parallel_for(ko::TeamThreadRange(mbr, 27), KOKKOS_LAMBDA (const Index& j) {
            const Index plut = 0;//ParentLUT::val(i,j);
            if ( neighbors(p,plut) != NULL_IND) {
                const Index h = neighbors(p, plut);
                //neighbors(t,j) = kids(h, ChildLUT::val(i,j));
            }
            else {
                neighbors(t,j) = NULL_IND;
            }
        });
    }
};

KOKKOS_INLINE_FUNCTION
Index nvertsAtLevel(const Int& lev) {
    Index points_per_edge = 2;
    for (Int l=1; l<=lev; ++l) {
        points_per_edge = 2*points_per_edge-1;
    }
    return cube(points_per_edge);
}


struct VertexOwnerFunctor {
    // output
    ko::View<Index*> owner;
    // input
    ko::View<key_type*> keys;
    ko::View<Index*[8]> neighbors;
    Int level;
    Int max_depth;
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const Index& i) const {
        
    }
    
};

}}
#endif
