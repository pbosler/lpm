#include "LpmNodeArrayInternal.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <bitset>

namespace Lpm {
namespace Octree {

std::string NodeArrayInternal::infoString() const {
    std::ostringstream ss;
    ss << "NodeArrayInternal (level " << level << " of " << max_depth << ") info:\n";
    ss << "\tnnodes = " << node_keys.extent(0) << "\n";
    auto keys = ko::create_mirror_view(node_keys);
    auto pt_start = ko::create_mirror_view(node_pt_idx);
    auto pt_ct = ko::create_mirror_view(node_pt_ct);
    auto prts = ko::create_mirror_view(node_parent);
    auto kids = ko::create_mirror_view(node_kids);
    auto rbox = ko::create_mirror_view(root_box);
    ko::deep_copy(keys, node_keys);
    ko::deep_copy(pt_start, node_pt_idx);
    ko::deep_copy(pt_ct, node_pt_ct);
    ko::deep_copy(prts, node_parent);
    ko::deep_copy(kids, node_kids);
    ko::deep_copy(rbox, root_box);
    
    ss << "\tNodes:\n";
    const Index nnodes = node_keys.extent(0);
    for (Index i=0; i<nnodes; ++i) {
        const BBox nbox = box_from_key(keys(i), rbox(), level, max_depth);
        ss << "\tnode(" << std::setw(8) << i << "): key = " << std::setw(8) << keys(i)
           << " " << std::bitset<3*MAX_OCTREE_DEPTH>(keys(i))
           << " pt_start = " << pt_start(i) << " pt_ct = " << pt_ct(i);
        if (level > 1) {
            ss << " parent = " << prts(i);
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
