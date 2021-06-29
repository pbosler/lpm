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
void Vertices<CoordsType>::insert_host(const Index crd_idx, const std::vector<Index>& edge_list) {
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
  for (int e=0; e<edge_list.size(); ++e) {
    edges(vert_idx,e) = edge_list[e];
  }
}

template <typename CoordsType>
std::string Vertices<CoordsType>::info_string(const int tab_level) const {
  std::ostringstream ss;
  auto tabstr = indent_string(tab_level);
  ss << tabstr << "Vertices info:\n";
  tabstr += "\t";
  ss << tabstr << "n_max() = " << n_max() << "\n";
  ss << tabstr << "nh() = " << nh() << "\n";
  ss << tabstr << "verts_are_dual() = " << std::boolalpha << verts_are_dual() << "\n";
  return ss.str();
}

// ETI
template class Vertices<Coords<PlaneGeometry>>;
template class Vertices<Coords<SphereGeometry>>;
template class Vertices<Coords<CircularPlaneGeometry>>;

}
