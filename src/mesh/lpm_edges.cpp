#include "lpm_edges.hpp"
#include <algorithm>
#include "util/lpm_floating_point_util.hpp"

#ifdef LPM_HAVE_NETCDF
#include "LpmNetCDF.hpp"
#endif

#include <sstream>

namespace Lpm {

void Edges::insert_host(const Index o, const Index d, const Index l, const Index r, const Index prt) {
  LPM_REQUIRE(_nmax >= _nh() + 1);
  const Index ins_pt = _nh();
  _ho(ins_pt) = o;
  _hd(ins_pt) = d;
  _hl(ins_pt) = l;
  _hr(ins_pt) = r;
  _hp(ins_pt) = prt;
  _hk(ins_pt, 0) = constants::NULL_IND;
  _hk(ins_pt, 1) = constants::NULL_IND;
  _nh() += 1;
  _hn_leaves() += 1;
}

#ifdef LPM_HAVE_NETCDF
Edges::Edges(const PolyMeshReader& reader) : Edges(reader.nEdges()) {
  reader.fill_origs(_ho);
  reader.fill_dests(_hd);
  reader.fill_lefts(_hl);
  reader.fill_rights(_hr);
  reader.fill_edge_tree(_hp, _hk, _hn_leaves());
  _nh() = origs.extent(0);
  updateDevice();
  }
#endif

template <typename SeedType>
void Edges::init_from_seed(const MeshSeed<SeedType>& seed) {
  LPM_REQUIRE(_nmax >= SeedType::nedges);
  for (Int i=0; i<SeedType::nedges; ++i) {
    _ho(i) = seed.seed_edges(i,0);
    _hd(i) = seed.seed_edges(i,1);
    _hl(i) = seed.seed_edges(i,2);
    _hr(i) = seed.seed_edges(i,3);
    _hp(i) = constants::NULL_IND;
    _hk(i,0) = constants::NULL_IND;
    _hk(i,1) = constants::NULL_IND;
  }
  _nh() = SeedType::nedges;
  _hn_leaves() = SeedType::nedges;
}

template <typename CoordsType>
void Edges::divide(const Index edge_idx, Vertices<CoordsType>& verts) {
  using Geo = typename CoordsType::crds_geometry_type;
  LPM_ASSERT(edge_idx < _nh());
  LPM_REQUIRE_MSG(_nh()+2 <= _nmax, "Edges::divide error, not enough memory.");
  LPM_ASSERT(!has_kids_host(edge_idx));

  // record state on entry
  const Index verts_insert_idx = verts.nh();
  const Index edges_insert_idx = _nh();

  // determine parent edge midpoint
  ko::View<Real[Geo::ndim],Host> midpt("midpt");
  ko::View<Real[Geo::ndim],Host> lag_midpt("lag_midpt");
  ko::View<Real[2][Geo::ndim],Host> endpts("parent_endpts");
  ko::View<Real[2][Geo::ndim],Host> lag_endpts("parent_lag_endpts");
  const Index orig_crd_idx = verts.crd_inds(_ho(edge_idx));
  const Index dest_crd_idx = verts.crd_inds(_hd(edge_idx));
  for (int i=0; i<Geo::ndim; ++i) {
    endpts(0,i) = verts.phys_crds->get_crd_component_host(orig_crd_idx, i);
    lag_endpts(0,i) = verts.lag_crds->get_crd_component_host(orig_crd_idx, i);
    endpts(1,i) = verts.phys_crds->get_crd_component_host(dest_crd_idx, i);
    lag_endpts(1,i) = verts.phys_crds->get_crd_component_host(dest_crd_idx, i);
  }
  Geo::midpoint(midpt, ko::subview(endpts, 0, ko::ALL), ko::subview(endpts, 1, ko::ALL));
  Geo::midpoint(lag_midpt, ko::subview(lag_endpts, 0, ko::ALL), ko::subview(lag_endpts, 1, ko::ALL));
  // insert new midpoint to vertices
  verts.insert_host(midpt, lag_midpt);
  // insert new child edges
  this->insert_host(_ho(edge_idx), verts_insert_idx, _hl(edge_idx), _hr(edge_idx), edge_idx);
  this->insert_host(verts_insert_idx, _hd(edge_idx), _hl(edge_idx), _hr(edge_idx), edge_idx);
  _hk(edge_idx, 0) = edges_insert_idx;
  _hk(edge_idx, 1) = edges_insert_idx+1;
  _hn_leaves()--;

  if (verts.verts_are_dual()) {
    // replace parent edge with children at vertices
    Int parent_idx_at_orig = constants::NULL_IND;
    Int parent_idx_at_dest = constants::NULL_IND;
    for (int e=0; e<constants::MAX_VERTEX_DEGREE; ++e) {
      if (verts.edges(_ho(edge_idx), e) == edge_idx) {
        parent_idx_at_orig = e;
      }
      if (verts.edges(_hd(edge_idx), e) == edge_idx) {
        parent_idx_at_dest = e;
      }
      if ( (parent_idx_at_orig != constants::NULL_IND) and
           (parent_idx_at_dest != constants::NULL_IND)) {
        break;
      }
    }

    LPM_ASSERT( (parent_idx_at_orig != constants::NULL_IND) and
                (parent_idx_at_dest != constants::NULL_IND) );

    verts.edges(_ho(edge_idx), parent_idx_at_orig) = edges_insert_idx;
    verts.edges(_hd(edge_idx), parent_idx_at_dest) = edges_insert_idx+1;

    // record edges at new vertex
    verts.edges(verts_insert_idx, 0) = edges_insert_idx;
    verts.edges(verts_insert_idx, 1) = edges_insert_idx+1;
  }

}



// template <> void Edges::divide<CircularPlaneGeometry>(const Index ind,
//   Coords<CircularPlaneGeometry>& crds, Coords<CircularPlaneGeometry>& lagcrds) {
//
//   assert(ind < _nh());
//
//   LPM_REQUIRE_MSG(_nh()+2 <= _nmax, "Edges::divide error: not enough memory.");
//   LPM_REQUIRE_MSG(!has_kids_host(ind), "Edges::divide error: called on previously divided edge.");
//
//   // record starting state
//   const Index crd_ins_pt = crds.nh();
//   const Index edge_ins_pt = _nh();
//
//   // determine edge midpoints
//   ko::View<Real[2], Host> midpt("midpt"), lagmidpt("lagmidpt");
//   ko::View<Real[2][2],Host> endpts("endpts"), lagendpts("lagendpts");
//   for (Short i=0; i<2; ++i) {
//     endpts(0,i) = crds.get_crd_component_host(_ho(ind), i);
//     lagendpts(0,i) = lagcrds.get_crd_component_host(_ho(ind),i);
//     endpts(1,i) = crds.get_crd_component_host(_hd(ind), i);
//     lagendpts(1,i) = lagcrds.get_crd_component_host(_hd(ind),i);
//   }
//   const Real lr0 = CircularPlaneGeometry::mag(ko::subview(lagendpts,0,ko::ALL()));
//   const Real lr1 = CircularPlaneGeometry::mag(ko::subview(lagendpts,1,ko::ALL()));
//   if (FloatingPoint<Real>::equiv(lr0, lr1, 10*constants::ZERO_TOL)) {
//     CircularPlaneGeometry::radial_midpoint(midpt, ko::subview(endpts, 0, ko::ALL()),
//       ko::subview(endpts,1,ko::ALL()));
//     CircularPlaneGeometry::radial_midpoint(lagmidpt, ko::subview(lagendpts, 0, ko::ALL()),
//       ko::subview(lagendpts,1,ko::ALL()));
//   }
//   else {
//     CircularPlaneGeometry::midpoint(midpt, ko::subview(endpts, 0, ko::ALL()),
//       ko::subview(endpts,1,ko::ALL()));
//     CircularPlaneGeometry::midpoint(lagmidpt, ko::subview(lagendpts, 0, ko::ALL()),
//       ko::subview(lagendpts,1,ko::ALL()));
//   }
//   crds.insert_host(midpt);
//   lagcrds.insert_host(lagmidpt);
//
//   insert_host(_ho(ind), crd_ins_pt, _hl(ind), _hr(ind), ind);
//   insert_host(crd_ins_pt, _hd(ind), _hl(ind), _hr(ind), ind);
//   _hk(ind,0) = edge_ins_pt;
//   _hk(ind,1) = edge_ins_pt+1;
//   _hn_leaves() -= 1;
// }

std::string Edges::info_string(const std::string& label, const short& tab_level, const bool& dump_all) const {
  std::ostringstream oss;
  const auto indent = indent_string(tab_level);

  oss << indent << "Edges " << label << " info: nh = (" << _nh() << ") of nmax = " << _nmax << " in memory; "
    << _hn_leaves() << " leaves." << std::endl;

  if (dump_all) {
    const auto bigindent = indent_string(tab_level+1);
    for (Index i=0; i<_nmax; ++i) {
      if (i==_nh()) oss << indent << "---------------------------------" << std::endl;
      oss << bigindent << label << ": (" << i << ") : ";
      oss << "orig = " << _ho(i) << ", dest = " << _hd(i);
      oss << ", left = " << _hl(i) << ", right = " << _hr(i);
      oss << ", parent = " << _hp(i) << ", kids = " << _hk(i,0) << "," << _hk(i,1);
      oss << std::endl;
    }
  }
  return oss.str();
}

/// ETI
template void Edges::divide<Coords<PlaneGeometry>>(const Index ind, Vertices<Coords<PlaneGeometry>>& verts);

template void Edges::divide<Coords<SphereGeometry>>(const Index ind, Vertices<Coords<SphereGeometry>>& verts);

template void Edges::init_from_seed(const MeshSeed<TriHexSeed>& seed);
template void Edges::init_from_seed(const MeshSeed<QuadRectSeed>& seed);
template void Edges::init_from_seed(const MeshSeed<CubedSphereSeed>& seed);
template void Edges::init_from_seed(const MeshSeed<IcosTriSphereSeed>& seed);
template void Edges::init_from_seed(const MeshSeed<UnitDiskSeed>& seed);

}
