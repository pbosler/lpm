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
    ko::View<ParentLUT> ptable;
    ko::View<ChildLUT> ctable;
    Int level; // assume > 0
    Int max_depth;
    
    NeighborhoodFunctor(ko::View<Index*[27]>& n, const ko::View<key_type*>& k, const ko::View<Index*[8]>& c,
        const ko::View<Index*>& p, const ko::View<Index*>& b, const Int& l, const Int& m) :
        neighbors(n), keys(k), kids(c), parents(p), base_address(b), level(l), max_depth(m), 
        ptable("ParentLUT"), ctable("ChildLUT") {}
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const member_type& mbr) const {
        const Index t = base_address(level) + mbr.league_rank();
        const Index p = parents(t);
        const key_type i = local_key(keys(t), level, max_depth);
        ko::parallel_for(ko::TeamThreadRange(mbr, 27), KOKKOS_LAMBDA (const Index& j) {
            const Index plut = table_val(i,j, ptable);
            if ( neighbors(p,plut) != NULL_IND) {
                const Index h = neighbors(p, plut);
                neighbors(t,j) = kids(h, table_val(i,j, ctable));
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
    ko::View<Index*[8]> owner;
    // input
    ko::View<key_type*> keys;
    ko::View<Index*[27]> neighbors;
    ko::View<NeighborsAtVertexLUT> nvtable;
    typedef typename ko::MinLoc<key_type,Index>::value_type minloc_type;
    
    VertexOwnerFunctor(ko::View<Index*[8]>& o, const ko::View<key_type*>& k, const ko::View<Index*[27]>& n) : 
        owner(o), keys(k), neighbors(n), nvtable("NeighborsAtVertexLUT") {}
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const member_type& mbr) const {
        const Index t = mbr.league_rank();  // each team is assigned a node
        ko::parallel_for(ko::TeamThreadRange(mbr,8), KOKKOS_LAMBDA (const Int& i) {
            // each thread gets a vertex of the team's node
            minloc_type result;
            // determine node that owns vertex v (the node sharing v with lowest key)
            ko::parallel_reduce(ko::ThreadVectorRange(mbr,8), [=] (const Int& j, minloc_type& loc) {
               const Index nbr_ind = neighbors(t, table_val(i,j,nvtable));
               if (nbr_ind != NULL_IND) {
                    if (keys(nbr_ind) < loc.val) {
                        loc.val = keys(nbr_ind);
                        loc.loc = nbr_ind;
                    }
               }
               }, ko::MinLoc<key_type,Index>(result));
            owner(t,i) = result.loc;
        });
    }
};

struct VertexFlagFunctor {
    // output
    ko::View<Index*[8]> flags;
    // input
    ko::View<Index*[8]> owner; 
    
    VertexFlagFunctor(ko::View<Index*[8]>& f, const ko::View<Index*[8]>& o) : flags(f), owner(o) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const member_type& mbr) const {
        const Index t = mbr.league_rank();
        ko::parallel_for(ko::TeamThreadRange(mbr, 8), KOKKOS_LAMBDA (const Int& i) {
            flags(t,i) = (owner(t,i) == t ? 1 : 0);
        });
    }
};

struct NVertsAtNodeFunctor {
    // output
    ko::View<Index*> nverts_at_node;
    // input
    ko::View<Index*[8]> flags;
    
    NVertsAtNodeFunctor(ko::View<Index*>& nv, const ko::View<Index*[8]>& f) : nverts_at_node(nv), flags(f) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const member_type& mbr) const {
        const Index t = mbr.league_rank();
        ko::parallel_reduce(ko::TeamThreadRange(mbr, 8), KOKKOS_LAMBDA (const Int& i, Int& ct) {
            ct += flags(t,i);
        }, nverts_at_node(t));
    }
};

struct VertexAddressFunctor {
    // output
    ko::View<Index*> vert_address;
    // input
    ko::View<Index*> nverts_at_node;
    
    VertexAddressFunctor(ko::View<Index*> va, const ko::View<Index*>& nv) : 
        vert_address(va), nverts_at_node(nv) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const Index& i, Index& ct, const bool& final_pass) const {
        const Index old_val = nverts_at_node(i);
        if (final_pass) {
            vert_address(i) = ct;
        }
        ct += old_val;
    }
};

struct VertexNodeFunctor {
    // output
    ko::View<Index*[8]> vert_nodes;
    ko::View<Index*[8]> node_verts;
    // input
    ko::View<Index*[8]> vert_owner;
    ko::View<Index*> vert_address;
    ko::View<Index*[27]> neighbors;
    ko::View<NeighborsAtVertexLUT> nvtable;

    VertexNodeFunctor(ko::View<Index*[8]>& vn, ko::View<Index*[8]>& nv, const ko::View<Index*[8]>& vo, const ko::View<Index*>& va,
        const ko::View<Index*[27]>& n) : vert_nodes(vn), node_verts(nv), vert_owner(vo), vert_address(va), neighbors(n),
        nvtable("NeighborsAtVertexLUT") {}
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const member_type& mbr) const {
        const Index t = mbr.league_rank();
        
        Int* local_vert_inds = (Int*)mbr.team_shmem().get_shmem(8*sizeof(Int));
        Int* local_scan = (Int*)mbr.team_shmem().get_shmem(8*sizeof(Int));
        
        ko::parallel_for(ko::TeamThreadRange(mbr, 8), KOKKOS_LAMBDA (const Int& i) {
            local_vert_inds[i] = (vert_owner(t,i) == t ? 1 : 0);
        });
        mbr.team_barrier();
        ko::parallel_for(ko::TeamThreadRange(mbr,8), KOKKOS_LAMBDA (const Int& i) {
            local_scan[i] = 0;
            for (Int j=0; j<=i; ++j) {
                local_scan[i] += local_vert_inds[j];
            }
        });
        mbr.team_barrier();
        ko::parallel_for(ko::TeamThreadRange(mbr, 8), KOKKOS_LAMBDA (const Int& i) {
            if (vert_owner(t,i) == t) {
                const Index v = vert_address(t) + local_scan[i];
                ko::parallel_for(ko::ThreadVectorRange(mbr,8), [=] (const Int& j) {
                    const Index nbr_ind = neighbors(t, table_val(i,j,nvtable));
                    vert_nodes(v,j) = nbr_ind;
                    node_verts(nbr_ind,j) = v;
                });
            }
        });
    }
    
    size_t team_shmem_size(int team_size) const {
        return 16*sizeof(Int);
    }
};


}}
#endif
