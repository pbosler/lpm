#ifndef LPM_VERTICES_HPP
#define LPM_VERTICES_HPP

#include "LpmConfig.h"
#include "lpm_kokkos_defs.hpp"
#include "lpm_constants.hpp"

#include <vector>
#include <memory>

namespace Lpm {


/** @brief class for recording information stored at vertices in a mesh with
vertices, edges, and faces.

This class is for topological information only.

*/
template <typename CoordsType>
class Vertices {
  public:
    n_view_type n;

    Vertices(const Index nmax, const bool verts_are_dual=false) :
        crd_inds("vert_crd_inds", nmax),
        n("n")
    {
      _nh = ko::create_mirror_view(n);
      _nh() = 0;
      _host_crd_inds = ko::create_mirror_view(crd_inds);
      if (verts_are_dual) {
        edges = ko::View<Index**, Host>("edges_at_vertex", nmax, constants::MAX_VERTEX_DEGREE);
        ko::deep_copy(edges, constants::NULL_IND);
      }
    }

    Vertices(const Index nmax, std::shared_ptr<CoordsType> pcrds,
        std::shared_ptr<CoordsType> lcrds);


    Index n_max() const {return crd_inds.extent(0); }

    Index nh() const {return _nh();}

    void update_device() const {
      phys_crds->update_device();
      lag_crds->update_device();
      ko::deep_copy(n, _nh);
    }

    void update_host() const {
      phys_crds->update_host();
      lag_crds->update_host();
      ko::deep_copy(_nh, n);
    }

    inline bool verts_are_dual() const {return edges.extent(0) > 0; }

    void insert_host(const Index crd_idx);

    void insert_host(const Index crd_idx, const std::vector<Index>& edge_list);

    template <typename PtViewType>
    void insert_host(const PtViewType physcrd, const PtViewType lagcrd) {
      const Index crd_insert_idx = phys_crds->nh();
      LPM_ASSERT(phys_crds);
      LPM_ASSERT(lag_crds);
      phys_crds->insert_host(physcrd);
      lag_crds->insert_host(lagcrd);
      this->insert_host(crd_insert_idx);
    }

    template <typename PtViewType>
    void insert_host(const PtViewType physcrd, const PtViewType lagcrd,
      const std::vector<Index>& edge_list) {
      LPM_ASSERT(phys_crds);
      LPM_ASSERT(lag_crds);
      const Index crd_insert_idx = phys_crds->nh();
      phys_crds->insert_host(physcrd);
      lag_crds->insert_host(lagcrd);
      this->insert_host(crd_insert_idx, edge_list);
    }

    void set_edges_host(const Index vert_idx, const std::vector<Index>& edge_list);

    index_view_type crd_inds;

    typename index_view_type::HostMirror host_crd_inds() const {return _host_crd_inds;}

    ko::View<Index**, Host> edges;

    std::shared_ptr<CoordsType> phys_crds;

    std::shared_ptr<CoordsType> lag_crds;

    std::string info_string(const std::string& label="", const int tab_level=0,
      const bool dump_all = false) const;

  private:
    typename n_view_type::HostMirror _nh;
    typename index_view_type::HostMirror _host_crd_inds;
};

}
#endif
