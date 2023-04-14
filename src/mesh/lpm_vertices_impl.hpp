#ifndef LPM_VERTICES_IMPL_HPP
#define LPM_VERTICES_IMPL_HPP

#include <sstream>
#include <string>

#include "lpm_assert.hpp"
#include "lpm_coords.hpp"
#include "lpm_coords_impl.hpp"
#include "mesh/lpm_vertices.hpp"
#include "util/lpm_string_util.hpp"

namespace Lpm {

template <typename CoordsType>
void Vertices<CoordsType>::insert_host(const Index crd_idx) {
  LPM_REQUIRE(n_max() >= _nh() + 1);
  _host_crd_inds(_nh()++) = crd_idx;
}

template <typename CoordsType>
template <typename PtViewType>
void Vertices<CoordsType>::insert_host(const Index crd_idx,
                                       const PtViewType& pcrd,
                                       const PtViewType& lcrd) {
  LPM_REQUIRE(n_max() >= _nh() + 1);
  _host_crd_inds(_nh()++) = crd_idx;
  phys_crds.insert_host(pcrd);
  lag_crds.insert_host(lcrd);
}

template <typename CoordsType>
template <typename MeshSeedType>
void Vertices<CoordsType>::init_from_seed(MeshSeedType& seed) {
  LPM_ASSERT(n_max() >= MeshSeedType::nverts);
  for (int i = 0; i < MeshSeedType::nverts; ++i) {
    const auto mcrd = Kokkos::subview(seed.seed_crds, i, Kokkos::ALL);
    this->insert_host(i, mcrd, mcrd);
  }
}

template <typename CoordsType>
Vertices<CoordsType>::Vertices(const Index nmax, CoordsType& pcrds,
                               CoordsType& lcrds)
    : crd_inds("vert_crd_inds", nmax),
      n("n"),
      phys_crds(pcrds),
      lag_crds(lcrds) {
  _nh = ko::create_mirror_view(n);
  _nh() = pcrds.nh();
  _host_crd_inds = ko::create_mirror_view(crd_inds);
  for (Index i = 0; i < pcrds.nh(); ++i) {
    _host_crd_inds(i) = i;
  }
  for (Index i = pcrds.nh(); i < nmax; ++i) {
    _host_crd_inds(i) = constants::NULL_IND;
  }
  ko::deep_copy(crd_inds, _host_crd_inds);
}

template <typename CoordsType>
void Vertices<CoordsType>::insert_host(const Index crd_idx,
                                       const std::vector<Index>& edge_list) {
  LPM_ASSERT(verts_are_dual());
  LPM_ASSERT(edge_list.size() <= edges.extent(1));
  LPM_REQUIRE(n_max() >= _nh() + 1);

  for (int e = 0; e < edge_list.size(); ++e) {
    edges(_nh(), e) = edge_list[e];
  }
  _host_crd_inds(_nh()++) = crd_idx;
}

template <typename CoordsType>
void Vertices<CoordsType>::set_edges_host(const Index vert_idx,
                                          const std::vector<Index>& edge_list) {
  LPM_ASSERT(vert_idx < _nh());
  LPM_ASSERT(edge_list.size() <= edges.extent(1));
  LPM_ASSERT(this->verts_are_dual());
  for (int e = 0; e < edge_list.size(); ++e) {
    edges(vert_idx, e) = edge_list[e];
  }
}

template <typename CoordsType>
std::string Vertices<CoordsType>::info_string(const std::string& label,
                                              const int tab_level,
                                              const bool dump_all) const {
  std::ostringstream ss;
  auto tabstr = indent_string(tab_level);
  ss << tabstr << "Vertices info (" << label << "):\n";
  tabstr += "\t";
  ss << tabstr << "n_max() = " << n_max() << "\n";
  ss << tabstr << "nh() = " << nh() << "\n";
  ss << tabstr << "verts_are_dual() = " << std::boolalpha << verts_are_dual()
     << "\n";
  ss << tabstr << phys_crds.info_string(label, tab_level + 1, dump_all);
  if (dump_all) {
    ss << lag_crds.info_string(label, tab_level + 1, dump_all);
  }
  return ss.str();
}

}  // namespace Lpm
#endif
