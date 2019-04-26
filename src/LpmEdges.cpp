#include "LpmEdges.hpp"
#include <sstream>

namespace Lpm {

void Edges::insertHost(const Index o, const Index d, const Index l, const Index r, const Index prt) {
    LPM_THROW_IF(_nmax < _nh(0) + 1, infoString("Edges::insertHost error: not enough memory."));
    const Index ins_pt = _nh(0);
    _ho(ins_pt) = o;
    _hd(ins_pt) = d;
    _hl(ins_pt) = l;
    _hr(ins_pt) = r;
    _hp(ins_pt) = prt;
    _hk(ins_pt, 0) = NULL_IND;
    _hk(ins_pt, 1) = NULL_IND;
    _nh(0) += 1;
    _hnLeaves(0) += 1;
}


template <typename SeedType>
void Edges::initFromSeed(const MeshSeed<SeedType>& seed) {
    LPM_THROW_IF(_nmax < SeedType::nedges, "Edges::initFromSeed error: not enough memory.");
    for (Int i=0; i<SeedType::nedges; ++i) {
        _ho(i) = seed.sedges(i,0);
        _hd(i) = seed.sedges(i,1);
        _hl(i) = seed.sedges(i,2);
        _hr(i) = seed.sedges(i,3);
        _hp(i) = NULL_IND;
        _hk(i,0) = NULL_IND;
        _hk(i,1) = NULL_IND;
    }
    _nh(0) = SeedType::nedges;
    _hnLeaves(0) = SeedType::nedges;
}

template <typename Geo> void Edges::divide(const Index ind, Coords<Geo>& crds, Coords<Geo>& lagcrds) {
    LPM_THROW_IF(_nh(0) + 2 > _nmax, "Edges::divide error: not enough memory.");
    LPM_THROW_IF(hasKidsHost(ind), "Edges::divide error: called on previously divided edge.");
    // record beginning state
    const Index crd_ins_pt = crds.nh();
    const Index edge_ins_pt = _nh(0);
    
    // determine edge midpoints
    ko::View<Real[Geo::ndim], Host> midpt("midpt"), lagmidpt("lagmidpt");
    ko::View<Real[2][Geo::ndim], Host> endpts("endpts"), lagendpts("lagendpts");
    for (int i=0; i<Geo::ndim; ++i) {
        endpts(0,i) = crds.getCrdComponentHost(_ho(ind), i);
        lagendpts(0,i) = lagcrds.getCrdComponentHost(_ho(ind), i);
        endpts(1,i) = crds.getCrdComponentHost(_hd(ind), i);
        lagendpts(1,i) = lagcrds.getCrdComponentHost(_hd(ind), i);
    }
    
    Geo::midpoint(midpt, ko::subview(endpts, 0, ko::ALL()), ko::subview(endpts, 1, ko::ALL()));
    Geo::midpoint(lagmidpt, ko::subview(lagendpts, 0, ko::ALL()), ko::subview(lagendpts, 1, ko::ALL()));
    // insert new midpoint to Coords 
    crds.insertHost(midpt);
    lagcrds.insertHost(lagmidpt);
    // insert new child edges
    insertHost(_ho(ind), crd_ins_pt, _hl(ind), _hr(ind), ind);
    insertHost(crd_ins_pt, _hd(ind), _hl(ind), _hr(ind), ind);
    _hk(ind,0) = edge_ins_pt;
    _hk(ind,1) = edge_ins_pt+1;    
    _hnLeaves(0) -= 1;
}

std::string Edges::infoString(const std::string& label) const {
    std::ostringstream oss;
    oss << "Edges " << label << " info: nh = (" << _nh(0) << ") of nmax = " << _nmax << " in memory; "
        << _hnLeaves(0) << " leaves." << std::endl;
    for (Index i=0; i<_nmax; ++i) {
        if (i==_nh(0)) oss << "---------------------------------" << std::endl;
        oss << label << ": (" << i << ") : ";
        oss << "orig = " << _ho(i) << ", dest = " << _hd(i);
        oss << ", left = " << _hl(i) << ", right = " << _hr(i);
        oss << ", parent = " << _hp(i) << ", kids = " << _hk(i,0) << "," << _hk(i,1);
        oss << std::endl;
    }
    return oss.str();
}

/// ETI
template void Edges::divide<PlaneGeometry>(const Index ind, Coords<PlaneGeometry>& crds, Coords<PlaneGeometry>& lagcrds);

template void Edges::divide<SphereGeometry>(const Index ind, Coords<SphereGeometry>& crds, Coords<SphereGeometry>& lagcrds);

template void Edges::initFromSeed(const MeshSeed<TriHexSeed>& seed);
template void Edges::initFromSeed(const MeshSeed<QuadRectSeed>& seed);
template void Edges::initFromSeed(const MeshSeed<CubedSphereSeed>& seed);
template void Edges::initFromSeed(const MeshSeed<IcosTriSphereSeed>& seed);

}
