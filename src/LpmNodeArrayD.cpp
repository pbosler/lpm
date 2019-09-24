#include "LpmNodeArrayD.hpp"
#include <string>
#include <iostream>
#include <iomanip>
#include <bitset>
#include <sstream>
#include "Kokkos_Core.hpp"

namespace Lpm {
namespace Octree {

std::string NodeArrayD::infoString() const {
    std::ostringstream ss;
    auto bv = ko::create_mirror_view(box);
    ko::deep_copy(bv, box);
    ss << "NodeArrayD info:\n";
    ss << "\tbounding box: " << bv();
    ss << "\tlevel = " << level << " of " << max_depth << " allowed.\n";
    
    auto keys = ko::create_mirror_view(node_keys);
    auto pt_start = ko::create_mirror_view(node_pt_idx);
    auto pt_ct = ko::create_mirror_view(node_pt_ct);
    auto prts = ko::create_mirror_view(node_parent);
    ko::deep_copy(keys, node_keys);
    ko::deep_copy(pt_start, node_pt_idx);
    ko::deep_copy(pt_ct, node_pt_ct);
    ko::deep_copy(prts, node_parent);
    ss << "\tNodes:\n";
    const Index nnodes = node_keys.extent(0);
    for (Index i=0; i<nnodes; ++i) {
        ss << "node(" << std::setw(8)<< i << "): key = " << std::setw(8) << keys(i) 
           << " " << std::bitset<MAX_OCTREE_DEPTH>(keys(i))
           << " pt_start = " << pt_start(i) << " pt_ct = " << pt_ct(i) 
           << " parent = " << node_parent(i) << "\n";
    }
    
    auto pin = ko::create_mirror_view(pt_in_node);
    auto oid = ko::create_mirror_view(orig_ids);
    ko::deep_copy(pin, pt_in_node);
    ko::deep_copy(oid, orig_ids);
    const Index npts = pts.extent(0);
    for (Index i=0; i<npts; ++i) {
        ss << "point(" << i << ") is in node " << pin(i) << " orig_id = " << oid(i) << "\n";
    }
    
    return ss.str();
}

}}
