#include "LpmOctree.hpp"
#include "LpmKokkosUtil.hpp"

namespace Lpm {

Index Octree::est_particles_per_leaf(const Index n, const Int max_depth) const {
    const Real dlam = std::sqrt(4*PI/n);
    const Real dx = 2.0/std::pow(2,max_depth);
    return Index(PARTICLE_MEMORY_PAD * std::ceil(dx/dlam));
}

Octree::Octree(const OneIndPerLeaf&, const Coords<SphereGeometry> crds) {
    const Index nleaves = crds.nh();
    const Int tree_depth = Int(std::ceil(std::log2(nleaves)/3));
    Index nnodes = 0;
    for (Int i=0; i<=tree_depth; ++i) {
        nnodes += std::pow(8,i);
    }
    nnodes += nleaves; // may need this extra space for unbalanced trees?
    boxes_ = bbox_type("bboxes", nnodes);
    kids_ = children_type("kids", nnodes);
    parent_ = parent_type("parent",nnodes);
    ko::parallel_for(nnodes, KOKKOS_LAMBDA (const Int& i) {
        if (i==0) {
            boxes_(i,0) = -1.1;
            boxes_(i,1) = 1.1;
            boxes_(i,2) = -1.1;
            boxes_(i,3) = 1.1;
            boxes_(i,4) = -1.1;
            boxes_(i,5) = 1.1;
        }
        for (int j=0; j<8; ++j)
            kids_(i,j) = NULL_IND;
        parent_(i) = NULL_IND;
    });
    
    auto policy = ExeSpaceUtils<>::get_default_team_policy(nleaves, WARP_SIZE);
}

}
