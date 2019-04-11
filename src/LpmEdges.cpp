#include "LpmEdges.hpp"
#include <sstream>

namespace Lpm {

template <typename Geo> void Edges::divide(const Index ind, Coords<Geo>& crds, Coords<Geo>& lagcrds) {
    // record beginning state
    const Index crd_ins_pt = crds.nh();
    const Index edge_ins_pt = _nh(0);
    
    // determine edge midpoints
    Real midpt[Geo::ndim];
    Real lagmidpt[Geo::ndim];
    Geo::midpoint(midpt, crds.getSlice(_ho(ind)), crds.getSlice(_hd(ind)));
    Geo::midpoint(lagmidpt, lagcrds.getSlice(_ho(ind)), lagcrds.getSlice(_hd(ind)));
    // insert new midpoint to Coords 
    crds.insertHost(midpt);
    lagcrds.insertHost(lagmidpt);
    // insert new child edges
    insertHost(_ho(ind), crd_ins_pt, _lefts(ind), _rights(ind), ind);
    insertHost(crd_ins_pt, _hd(ind), _lefts(ind), _rights(ind), ind);
    _hk(ind,0) = edge_ins_pt;
    _hk(ind,1) = edge_ins_pt+1;    
}

void Edges::printedges(const std::string& label) const {
    std::ostringstream oss;
    for (Index i=0; i<_nmax; ++i) {
        oss << label << ": (" << i << ") : ";
        oss << "orig = " << _ho(i) << ", dest = " << _hd(i);
        oss << ", left = " << _hl(i) << ", right = " << _hr(i);
        oss << ", parent = " << _hp(i) << ", kids = " << _hk(i,0) << "," << _hk(i,1);
        oss << std::endl;
        std::cout << oss.str();
        oss.str("");
    }
}

/// ETI
template void Edges::divide<PlaneGeometry>(const Index ind, Coords<PlaneGeometry>& crds, Coords<PlaneGeometry>& lagcrds);

template void Edges::divide<SphereGeometry>(const Index ind, Coords<SphereGeometry>& crds, Coords<SphereGeometry>& lagcrds);

}
