#ifndef LPM_OCTREE_HPP
#define LPM_OCTREE_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmCoords.hpp"
#include "Kokkos_Core.hpp"
#include <cmath>

namespace Lpm {

template <typename CVT3, typename CVT6> KOKKOS_INLINE_FUNCTION
Int local_child_index(const CVT3 pos, const CVT6 box) {
    Int result = 0;
    if (pos(0) > 0.5*(box(0) + box(1))) result += 1;
    if (pos(1) > 0.5*(box(2) + box(3))) result += 2;
    if (pos(2) > 0.5*(box(4) + box(5))) result += 4;
    return result;
}

template <typename VT6> KOKKOS_INLINE_FUNCTION
void divide_box(VT6 boxes, const Index parent_ind, const Index kid0) {
    auto pbox = ko::subview(boxes, parent_ind, ko::ALL());
    const Real xmin = pbox(0);
    const Real xmax = pbox(1);
    const Real ymin = pbox(2);
    const Real ymax = pbox(3);
    const Real zmin = pbox(4);
    const Real zmax = pbox(5);
    const Real xmid = 0.5*(xmin + xmax);
    const Real ymid = 0.5*(ymin + ymax);
    const Real zmid = 0.5*(zmin + zmax);
    for (int j=0; j<8; ++j) {
        boxes(kid0+j,0) = (j%2 == 0 ? xmin : xmid);
        boxes(kid0+j,1) = (j%2 == 0 ? xmid : xmax);
        boxes(kid0+j,2) = ((j>>1)&1 == 0 ? ymin : ymid);
        boxes(kid0+j,3) = ((j>>1)&1 == 0 ? ymid : ymax);
        boxes(kid0+j,4) = (j>>2 == 0 ? zmin : zmid);
        boxes(kid0+j,5) = (j>>2 == 0 ? zmid : zmax); 
    }
}


struct OneIndPerLeaf {
    /**
        Use double-ended arrays as in Burtscher & Pingali's Barnes Hut algorithm
        
        nnodes = n_internal + nleaves
        array size N >= nnodes
        Array indices 0, 1, 2, ..., n_internal-1, are internal nodes
        N-nleaves >= n_internal
        Array indices N-nleaves, ..., N-1, are leaves

    */
    struct BuildTree {
        ko::View<Real*[6]> boxes;
        ko::View<Index*[8]> kids;
        ko::View<Index*> parents;
        ko::View<const Real*[3]> crds;
        Index nn;
        
        BuildTree(ko::View<Real*[6]> bb, ko::View<Index*[8]> kk, ko::View<Index*> pp, 
            const ko::View<const Real*[3]> cc) : boxes(bb), kids(kk), parents(pp), crds(cc), 
                nn(1) {}
    
        void operator (member_type member) const {
            const Index n_particles = member.league_size();
            const Int ind_inc = member.team_size();
            Index work_index = member.league_rank()*member.team_size() + member.team_rank();
            Int do_particle_flag = 1;
            Index parent_ind = 0;
            Index kid_ind = 0;
            Int loc_ind = 0;
            while (work_index > 0 && work_index < n_particles) {
                auto pos = ko::subview(crds, work_index, ko::ALL());
//                 if (do_particle_flag != 0) {
                    // new particle -- start at root
//                     do_particle_flag = 0;
                    parent_ind = 0;
                    depth = 1;
                    auto parent_box = ko::subview(boxes, parent_ind, ko::ALL());
                    kid_loc = local_child_index(pos, parent_box);
                    node_ind = kids(parent_ind, kid_loc);
//                 }
                /** follow tree to leaf
                    kid_ind <= 0 means leaf node is empty
                    kid_ind >= nn means means leaf node is already full --- division required
                */
                while (node_ind > 0 && node_ind < nn) {
                    parent_ind = kid_ind;
                    auto parent_box = ko::subview(boxes, parent_ind, ko::ALL());
                    loc_ind = local_child_index(pos, parent_box);
                    kid_ind = kids(parent_ind, loc_ind);
                    depth++;
                }
                
                // case 0: leaf is locked.  try again later.
                if (kid_ind != LOCK_IND) {
                    if (kid_ind == NULL_IND) {
                        /** case 1: kid_ind = NULL_IND
                            leaf is empty; add particle and continue
                        */
                        if (ko::atomic_compare_exchange(kids(parent_ind,loc_ind), NULL_IND,
                             kids.extent(0)-work_index-1)) {
                            parents(kids(parent_ind,loc_ind)) = parent_ind;
                            work_index += ind_inc;            
                        }
                    }
                    else {
                    /** case 2: kid_ind > nn
                        leaf is full; divide leaf
                        ---- divide leaf ----
                        acquire lock
                        reserve space for children
                        create children
                        add existing particle to appropriate child
                        case 2a: new particle is not in same new child as existing particle
                            add to appropriate child
                        case 2b: new particle is in same new child as existing particle
                            divide that child
                            add both particles to its kids
                    */
                        do {
                            depth++;
                            if (depth > MAX_DEPTH) {
                                // Error: max depth exceeded.  Particles have collided.
                            }
                            if (nn+8 > kids.extent(0)-n_particles) {
                                // Error: not enough memory.
                            }
                        } while () {}
                    }
                }                
            }
    };
};
struct NIndsPerLeaf {};

class Octree {
    public:
        typedef ko::View<Real*[6],Dev> bbox_type;
        typedef ko::View<Index*[8],Dev> children_type;
        typedef ko::View<Index*,Dev> parent_type;        
        typedef ko::View<Index**,Dev> particle_index_type;

        Octree(const OneIndPerLeaf&, const Coords<SphereGeometry>& crds);
        
    protected:
        static constexpr Int PARTICLE_MEMORY_PAD = 10;
        static constexpr Int MAX_DEPTH = 32;
        static constexpr Int BLOCK_SIZE = 1024;
        static constexpr Int WARP_SIZE = 32;
    
        bbox_type boxes_;
        children_type kids_;
        parent_type parent_;
        particle_index_type inds_;
        Index nNodes_;
        Index nLeaves_;
        
        void initNull();
        
        Index est_particles_per_leaf(const Index n, const Int max_depth) const;        

//         KOKKOS_INLINE_FUNCTION
//         Index nNodes(const Int max_depth) const {
//             Index result = 0;
//             for (Int i=0; i<=max_depth) result += std::pow(8,i);
//             return result;
//         };
//         
//         KOKKOS_INLINE_FUNCTION
//         Index nLeaves(const Int max_depth) const {
//             return std::pow(8,max_depth);
//         };
//         
//         KOKKOS_INLINE_FUNCTION
//         Index nInternalNodes(const Int max_depth) const {
//             return nNodes(max_depth) - nLeaves(max_depth);
//         };
        
//         void initMaxIndsPerLeaf(const Int nmax);
        
//         void initOneIndPerLeaf();
};


}
#endif
