#include "LpmNodeArrayInternal.hpp"
#include "LpmOctreeKernels.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <bitset>

namespace Lpm {
namespace Octree {

void NodeArrayInternal::initFromLeaves(NodeArrayD& leaves) {
    const Index nparents = leaves.node_keys.extent(0)/8;
    std::cout << "nparents = " << nparents << "\n";
    ko::View<key_type*> pkeys("parent_keys", nparents);
    ko::View<Index*[2]> pinds("parent_inds", nparents);
    std::cout << "pkeys, pinds allocated.\n";
    auto parent_policy = ExeSpaceUtils<>::get_default_team_policy(nparents, 8);
    std::cout << "starting ParentNodeFunctor ||for\n";
    ko::parallel_for(parent_policy, ParentNodeFunctor(pkeys, pinds, leaves.node_keys, leaves.node_pt_inds,
        level, max_depth));
    std::cout << "parent node functor pfor returned.\n";
}


std::string NodeArrayInternal::infoString() const {
    std::ostringstream ss;
    ss << "NodeArrayInternal (level " << level << " of " << max_depth << ") info:\n";
    ss << "\tnnodes = " << node_keys.extent(0) << "\n";
    auto keys = ko::create_mirror_view(node_keys);
    auto pt_inds = ko::create_mirror_view(node_pt_inds);
    auto parents = ko::create_mirror_view(node_parents);
    auto kids = ko::create_mirror_view(node_kids);
    auto rbox = ko::create_mirror_view(root_box);
    ko::deep_copy(keys, node_keys);
    ko::deep_copy(pt_inds, node_pt_inds);
    ko::deep_copy(parents, node_parents);
    ko::deep_copy(kids, node_kids);
    ko::deep_copy(rbox, root_box);
    
    ss << "\tNodes:\n";
    const Index nnodes = node_keys.extent(0);
    for (Index i=0; i<nnodes; ++i) {
        const BBox nbox = box_from_key(keys(i), rbox(), level, max_depth);
        ss << "\tnode(" << std::setw(8) << i << "): key = " << std::setw(8) << keys(i)
           << " " << std::bitset<3*MAX_OCTREE_DEPTH>(keys(i))
           << " pt_start = " << pt_inds(i,0) << " pt_ct = " << pt_inds(i,1);
        if (level > 1) {
            ss << " parent = " << parents(i);
        }
        ss << " kids: ";
        for (int j=0; j<8; ++j) {
            ss << kids(i,j) << " ";
        }
        ss << "\n";
        ss << "\tbox = " << nbox << "\tedge_len <= " << longestEdge(nbox) << " ar = " << boxAspectRatio(nbox) << "\n";
    }
    return ss.str();
}

}}
