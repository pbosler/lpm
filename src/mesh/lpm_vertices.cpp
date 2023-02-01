#include "lpm_vertices.hpp"
#include "lpm_assert.hpp"
#include "lpm_coords.hpp"
#include "util/lpm_string_util.hpp"
#include <string>
#include <sstream>

namespace Lpm {

template <typename CoordsType>
void Vertices<CoordsType>::insert_host(const Index crd_idx) {
  LPM_REQUIRE(n_max() >= _nh()+1);
  _host_crd_inds(_nh()++) = crd_idx;
}

template <typename CoordsType>
Vertices<CoordsType>::Vertices(const Index nmax, std::shared_ptr<CoordsType> pcrds,
  std::shared_ptr<CoordsType> lcrds) :
    crd_inds("vert_crd_inds", nmax),
    n("n"),
    phys_crds(pcrds),
    lag_crds(lcrds) {
    _nh = ko::create_mirror_view(n);
    _nh() = pcrds->nh();
    _host_crd_inds = ko::create_mirror_view(crd_inds);
    for (Index i=0; i<pcrds->nh(); ++i) {
      _host_crd_inds(i) = i;
    }
    for (Index i=pcrds->nh(); i<nmax; ++i) {
      _host_crd_inds(i) = constants::NULL_IND;
    }
    ko::deep_copy(crd_inds, _host_crd_inds);
}


template <typename CoordsType>
void Vertices<CoordsType>::insert_host(const Index crd_idx, const std::vector<Index>& edge_list) {
  LPM_ASSERT(verts_are_dual());
  LPM_ASSERT(edge_list.size() <= edges.extent(1));
  LPM_REQUIRE(n_max() >= _nh()+1);

  for (int e=0; e<edge_list.size(); ++e) {
    edges(_nh(), e) = edge_list[e];
  }
  _host_crd_inds(_nh()++) = crd_idx;
}

template <typename CoordsType>
void Vertices<CoordsType>::set_edges_host(const Index vert_idx, const std::vector<Index>& edge_list) {
  LPM_ASSERT(vert_idx < _nh());
  LPM_ASSERT(edge_list.size() <= edges.extent(1));
  LPM_ASSERT(this->verts_are_dual());
  for (int e=0; e<edge_list.size(); ++e) {
    edges(vert_idx,e) = edge_list[e];
  }
}

template <typename CoordsType>
std::string Vertices<CoordsType>::info_string(const std::string& label, const int tab_level,
  const bool dump_all) const {
  std::ostringstream ss;
  auto tabstr = indent_string(tab_level);
  ss << tabstr << "Vertices info (" << label << "):\n";
  tabstr += "\t";
  ss << tabstr << "n_max() = " << n_max() << "\n";
  ss << tabstr << "nh() = " << nh() << "\n";
  ss << tabstr << "verts_are_dual() = " << std::boolalpha << verts_are_dual() << "\n";
  if (phys_crds) {
    ss << tabstr << phys_crds->info_string(label, tab_level+1, dump_all);
  }
  else {
    ss << tabstr << "phys_crds = NULL\n";
  }
  if (dump_all) {
    if (lag_crds) {
      ss << lag_crds->info_string(label, tab_level+1, dump_all);
    }
    else {
      ss << tabstr << "lag_crds = NULL\n";
    }
  }
  return ss.str();
}

// ETI
template class Vertices<Coords<PlaneGeometry>>;
template class Vertices<Coords<SphereGeometry>>;
// template class Vertices<Coords<CircularPlaneGeometry>>;

}
