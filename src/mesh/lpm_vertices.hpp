#ifndef LPM_VERTICES_HPP
#define LPM_VERTICES_HPP

#include <memory>
#include <vector>

#include "LpmConfig.h"
#include "lpm_constants.hpp"
#include "lpm_kokkos_defs.hpp"

namespace Lpm {

#ifdef LPM_USE_NETCDF
template <typename Geo>
class NcWriter;
class PolymeshReader;
#endif

/** @brief class for recording information stored at vertices in a mesh with
vertices, edges, and faces.

This class is for topological information only.

*/
template <typename CoordsType>
class Vertices {
 public:
  /** @brief Number of vertices currently used.

    n <= nmax, where nmax is the maximum allowed in memory.
  */
  n_view_type n;

  /** @brief Constructor with no associated Coords.

    @param [in] nmax maximum number of vertices; used to allocate memory
    @param [in] verts_are_dual if true, memory is allocated to store indices
                of edges incident to each vertex.
  */
  Vertices(const Index nmax, const bool verts_are_dual = false)
      : crd_inds("vert_crd_inds", nmax), n("n") {
    _nh = ko::create_mirror_view(n);
    _nh() = 0;
    _host_crd_inds = ko::create_mirror_view(crd_inds);
    if (verts_are_dual) {
      edges = ko::View<Index**>("edges_at_vertex", nmax,
                                constants::MAX_VERTEX_DEGREE);
      _host_edges = Kokkos::create_mirror_view(edges);
      ko::deep_copy(edges, constants::NULL_IND);
      ko::deep_copy(_host_edges, edges);
    }
  }

  template <typename MeshSeedType>
  friend class PolyMesh2d;

#ifdef LPM_USE_NETCDF
  template <typename Geo>
  friend class NcWriter;

  friend class PolymeshReader;
#endif

  /** @brief Constructor with associated Coords.

    @todo nmax parameter is redundant, can be taken from coords.

    @param [in] nmax maximum number of vertices; used to allocate memory
    @param [in] pcrds pointer to physical coordinates
    @param [in] lcrds pointer to Lagrangian coordinates
  */
  Vertices(const Index nmax, std::shared_ptr<CoordsType> pcrds,
           std::shared_ptr<CoordsType> lcrds);

  /** @brief return the maximum allowed number of vertices in memory

    @hostfn
  */
  Index n_max() const { return crd_inds.extent(0); }

  /** @brief  Return the current number of vertices

    @hostfn
  */
  Index nh() const { return _nh(); }

  /** @brief  deep-copy all data from host to device
   */
  void update_device() const {
    phys_crds->update_device();
    lag_crds->update_device();
    ko::deep_copy(n, _nh);
    if (verts_are_dual()) {
      ko::deep_copy(edges, _host_edges);
    }
  }

  /** @brief deep-copy all data from device to host
   */
  void update_host() const {
    phys_crds->update_host();
    lag_crds->update_host();
    ko::deep_copy(_nh, n);
    if (verts_are_dual()) {
      ko::deep_copy(_host_edges, edges);
    }
  }

  /** @brief True if memory is allocated to store edges incident to each vertex
   */
  inline bool verts_are_dual() const { return edges.extent(0) > 0; }

  /** @brief Insert a new vertex.

    @hostfn

    Inserts a vertex whose associated coordinates are at crd_idx in the
    physical and Lagrangian Coords arrays.

    @param [in] crd_idx index of coordinates associated with the new vertex
  */
  void insert_host(const Index crd_idx);

  /** @brief Insert a new vertex

    @hostfn

    Inserts a vertex whose associated coordinates are at crd_index in the
    physical and Lagrangian coordinate arrays, with edge data for dual meshes.

    @param [in] crd_idx index of coordinates associated with the new vertex
    @param [in] edge_list list of edges incident to new vertex
  */
  void insert_host(const Index crd_idx, const std::vector<Index>& edge_list);

  /** @brief  Insert a new vertex

    Inserts a vertex at the given coordinates.  Inserts the coordinates, too,
    both at the end of the current arrays.

    @hostfn

    @param [in] physcrd physical coordinates of a single point (the vertex to
    add)
    @param [in] lagcrd Lagrangian coordinates of the vertex to add
  */
  template <typename PtViewType>
  void insert_host(const PtViewType physcrd, const PtViewType lagcrd) {
    const Index crd_insert_idx = phys_crds->nh();
    LPM_ASSERT(phys_crds);
    LPM_ASSERT(lag_crds);
    phys_crds->insert_host(physcrd);
    lag_crds->insert_host(lagcrd);
    this->insert_host(crd_insert_idx);
  }

  /** @brief Insert a new vertex

    Inserts a vertex at the given coordinates with dual mesh edges.
    Inserts the coordinates, too, both at the end of the current arrays.

    @hostfn

    @param [in] physcrd physical coordinates of a single point (the vertex to
    add)
    @param [in] lagcrd Lagrangian coordinates of the vertex to add
    @param [in] edge_list list of edges incident to the new vertex
  */
  template <typename PtViewType>
  void insert_host(const PtViewType physcrd, const PtViewType lagcrd,
                   const std::vector<Index>& edge_list) {
    LPM_ASSERT(phys_crds);
    LPM_ASSERT(lag_crds);
    LPM_ASSERT(verts_are_dual());
    const Index crd_insert_idx = phys_crds->nh();
    phys_crds->insert_host(physcrd);
    lag_crds->insert_host(lagcrd);
    this->insert_host(crd_insert_idx, edge_list);
  }

  /** @brief Set the edges incident to a vertex.

    @hostfn

    @param [in] vert_idx index of vertex to update
    @param [in] indices of incident edges
  */
  void set_edges_host(const Index vert_idx,
                      const std::vector<Index>& edge_list);

  /** @brief Indices of coordinates in a Coords object

    vertex i has physical coordinates at crd_inds(i) of phys_crds.
  */
  index_view_type crd_inds;

  /** @brief Host mirror of crd_inds.
   */
  typename index_view_type::HostMirror host_crd_inds() const {
    return _host_crd_inds;
  }

  /** Return the index of the coordinates of a vertex

    @hostfn

    @param [in] i vertex Index
    @return coordinate index
  */
  Index host_crd_ind(const Index i) const { return _host_crd_inds(i); }

  /** @brief Array to hold indices of edges incident to each vertex. Only used
    for dual meshes.  If mesh is not dual, this view is not allocated.
  */
  ko::View<Index**> edges;

  /** @brief Smart pointer to a Coords object containing the physical
    coordinates of each vertex.
  */
  std::shared_ptr<CoordsType> phys_crds;

  /** @brief Smart pointer to a Coords object containing the Lagrangian
    coordinates of each vertex.
  */
  std::shared_ptr<CoordsType> lag_crds;

  /** @brief Create and return a string containing info about this Vertices
    instance.

    Used for logging.

    @hostfn

    @param [in] label : label for contained info
    @param [in] tab_level : indentation level for string
    @param [in] dump_all : if true, all vertex data will be written to string
  */
  std::string info_string(const std::string& label = "",
                          const int tab_level = 0,
                          const bool dump_all = false) const;

 private:
  /// Host mirror of view containing the number of active vertices
  typename n_view_type::HostMirror _nh;
  /// Host mirror of view containing each vertex's coordinate indices
  typename index_view_type::HostMirror _host_crd_inds;
  /// Host mirror of view containing each vertices incident edges
  /// Only used in dual meshes.
  typename Kokkos::View<Index**>::HostMirror _host_edges;
};

}  // namespace Lpm
#endif
