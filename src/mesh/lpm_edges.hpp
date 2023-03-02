#ifndef LPM_EDGES_HPP
#define LPM_EDGES_HPP

#include "Kokkos_Core.hpp"
#include "LpmConfig.h"
#include "lpm_assert.hpp"
#include "lpm_constants.hpp"
#include "lpm_coords.hpp"
#include "lpm_geometry.hpp"
#include "lpm_mesh_seed.hpp"
#include "mesh/lpm_vertices.hpp"
#include "util/lpm_string_util.hpp"

namespace Lpm {

#ifdef LPM_USE_NETCDF
template <typename Geo>
class NcWriter;
class PolymeshReader;
#endif

/** @brief Edges of panels connect Coords to Faces

  Each edge has an origin vertex and a destination vertex (in Coords), a left
  face and a right face (in Faces).

  This class stores arrays that define a collection of edges.

  @warning  Modifications are only allowed on host.
  All device functions are read-only and const.
*/
class Edges {
 public:
  typedef ko::View<Index*> edge_view_type;  ///< type to hold pointers (array
                                            ///< indices) to other mesh objects
  typedef typename edge_view_type::HostMirror edge_host_type;
  typedef ko::View<Index* [2]>
      edge_tree_view;  ///< Edge division results in a binary tree; this holds
                       ///< its data
  typedef typename edge_tree_view::HostMirror edge_tree_host;

  template <typename MeshSeedType>
  friend class PolyMesh2d;
#ifdef LPM_USE_NETCDF
  template <typename Geo>
  friend class NcWriter;

  friend class PolymeshReader;

  explicit Edges(const PolymeshReader& reader);
#endif

  edge_view_type origs;   ///< pointers to edge origin vertices
  edge_view_type dests;   ///< pointers to edge destination vertices
  edge_view_type lefts;   ///< pointers to edges' left faces
  edge_view_type rights;  ///< pointers to edges' right faces
  edge_view_type parent;  ///< pointers to parent edges
  edge_tree_view kids;    ///< pointers to child edges
  n_view_type n;          ///< number of initialized edges
  n_view_type n_leaves;   ///< number of leaf edges

  /** @brief Constructor.

    @param nmax Maximum number of edges to allocate space.
    @see MeshSeed::setMaxAllocations()
  */
  explicit Edges(const Index nmax)
      : origs("origs", nmax),
        dests("dests", nmax),
        lefts("lefts", nmax),
        rights("rights", nmax),
        parent("parent", nmax),
        kids("kids", nmax),
        n("n"),
        _nmax(nmax),
        n_leaves("nLeaves") {
    _nh = ko::create_mirror_view(n);
    _ho = ko::create_mirror_view(origs);
    _hd = ko::create_mirror_view(dests);
    _hl = ko::create_mirror_view(lefts);
    _hr = ko::create_mirror_view(rights);
    _hp = ko::create_mirror_view(parent);
    _hk = ko::create_mirror_view(kids);
    _hn_leaves = ko::create_mirror_view(n_leaves);
    _nh() = 0;
    _hn_leaves() = 0;
  }

  /** \brief Copies edge data from host to device.
   */
  void update_device() const {
    ko::deep_copy(origs, _ho);
    ko::deep_copy(dests, _hd);
    ko::deep_copy(lefts, _hl);
    ko::deep_copy(rights, _hr);
    ko::deep_copy(parent, _hp);
    ko::deep_copy(kids, _hk);
    ko::deep_copy(n, _nh);
    ko::deep_copy(n_leaves, _hn_leaves);
  }

  /** @brief Copies edge data from device to host.

   currently, edges are modified only the host (for AMR),
   so this function is unnecessary.  Future impls may include
   mesh handling on the device.
  */
  void update_host() const {}

  /** @brief Return a vector pointing from an edge's origin to its destination

    @param [in/out] evec edge vector
    @param [in] vcrds vertex coordinates
    @param [in] e_idx index of edge
  */
  template <typename V, typename VertCrds>
  KOKKOS_INLINE_FUNCTION void edge_vector(V& evec, const VertCrds& vcrds,
                                          const Index e_idx) const {
    const auto dest_crd = Kokkos::subview(vcrds, dests[e_idx], Kokkos::ALL);
    const auto orig_crd = Kokkos::subview(vcrds, origs[e_idx], Kokkos::ALL);
    for (int i = 0; i < dest_crd.extent(0); ++i) {
      evec[i] = dest_crd[i] - orig_crd[i];
    }
  }

  /** \brief Returns true if the edge is on the boundary of the domain

  */
  KOKKOS_INLINE_FUNCTION
  bool on_boundary(const Index ind) const {
    LPM_KERNEL_ASSERT(ind < n());
    return lefts(ind) == constants::NULL_IND ||
           rights(ind) == constants::NULL_IND;
  }

  /** \brief Returns true if the edge has been divided.

  */
  KOKKOS_INLINE_FUNCTION
  bool has_kids(const Index ind) const {
    LPM_KERNEL_ASSERT(ind < n());
    return ind < n() && kids(ind, 0) > 0;
  }

  /** \brief Maximum number of edges allowed in memory
   */
  KOKKOS_INLINE_FUNCTION
  Index n_max() const { return origs.extent(0); }

  /** Number of initialized edges.

  \hostfn
  */
  inline Index nh() const { return _nh(); }

  /** Inserts a new edge into the data structure.

  \hostfn

  \param o index of origin vertex for new edge
  \param d index of destination vertex for new edge
  \param l index of left panel for new edge
  \param r index of right panel for new edge
  \param prt index of parent edge
  */
  void insert_host(const Index o, const Index d, const Index l, const Index r,
                   const Index prt = constants::NULL_IND);

  /** \brief Divides an edge, creating two children

  \hostfn


  Child edges have same left/right panels as parent edge.

  The first child, whose index will be n is the 0th index child of the parent,
  and shares its origin vertex. A new midpoint is added to both sets of Coords.
  The second child has index n+1 is the 1th index child of the parent, and
  shares its destination vertex.

  @todo LPM_ASSERT(n==child(ind,0))
  @todo LPM_ASSERT(n+1==child(ind,1))
  @todo LPM_ASSERT(orig(n) == orig(ind))
  @todo LPM_ASSERT(dest(n+1) == dest(ind))

  \param ind index of edge to be divided
  \param crds physical coordinates of edge vertices
  \param lagcrds Lagrangian coordinates of edge vertices
  */
  template <typename CoordsType>
  void divide(const Index edge_idx, Vertices<CoordsType>& verts);

  /** \brief Overwrite the left panel of an edge

    \hostfn

    \param ind edge whose face needs updating
    \param newleft new index of the edge's left panel
  */
  inline void set_left(const Index ind, const Index newleft) {
    LPM_ASSERT(ind < _nh());
    _hl(ind) = newleft;
  }

  /** \brief Overwrite the right panel of an edge

    \hostfn

    \param ind edge whose face needs updating
    \param newright new index of the edge's right panel
  */
  inline void set_right(const Index ind, const Index newright) {
    LPM_ASSERT(ind < _nh());
    _hr(ind) = newright;
  }

  inline void set_orig(const Index ind, const Index& neworig) {
    LPM_ASSERT(ind < _nh());
    _ho(ind) = neworig;
  }

  inline void set_dest(const Index ind, const Index& newdest) {
    LPM_ASSERT(ind < _nh());
    _hd(ind) = newdest;
  }

  /** Initialize a set of Edges from a MeshSeed

  \hostfn

  \param seed MeshSeed
  */
  template <typename SeedType>
  void init_from_seed(const MeshSeed<SeedType>& seed);

  /** Return the requested child (0 or 1) of an edge.

  \hostfn

  \param ind index of edge whose child is needed
  \param child either 0 or 1, to represent the first or second child
  \see divide()
  */
  inline Index kid_host(const Index ind, const Int child) const {
    LPM_ASSERT(ind < _nh());
    return _hk(ind, child);
  }

  /// Host function
  std::string info_string(const std::string& label = "",
                          const short& tab_level = 0,
                          const bool& dump_all = false) const;

  /// Host functions
  inline Index orig_host(const Index ind) const {
    LPM_ASSERT(ind < _nh());
    return _ho(ind);
  }
  inline Index dest_host(const Index ind) const {
    LPM_ASSERT(ind < _nh());
    return _hd(ind);
  }
  inline Index left_host(const Index ind) const {
    LPM_ASSERT(ind < _nh());
    return _hl(ind);
  }
  inline Index right_host(const Index ind) const {
    LPM_ASSERT(ind < _nh());
    return _hr(ind);
  }
  inline Index parent_host(const Index ind) const {
    LPM_ASSERT(ind < _nh());
    return _hp(ind);
  }

  inline edge_host_type origs_host() const { return _ho; }
  inline edge_host_type dests_host() const { return _hd; }
  inline edge_host_type lefts_host() const { return _hl; }
  inline edge_host_type rights_host() const { return _hr; }
  inline edge_host_type parents_host() const { return _hp; }
  inline edge_tree_host kids_host() const { return _hk; }

  /// Host function
  inline bool on_boundary_host(const Index ind) const {
    LPM_ASSERT(ind < _nh());
    return _hl(ind) == constants::NULL_IND || _hr(ind) == constants::NULL_IND;
  }

  /// Host function
  inline bool has_kids_host(const Index ind) const {
    LPM_ASSERT(ind < _nh());
    return ind < _nh() && _hk(ind, 0) > 0;
  }

  inline Index n_leaves_host() const { return _hn_leaves(); }

 protected:
  edge_host_type _ho;
  edge_host_type _hd;
  edge_host_type _hl;
  edge_host_type _hr;
  edge_host_type _hp;
  edge_tree_host _hk;
  ko::View<Index>::HostMirror _nh;
  ko::View<Index>::HostMirror _hn_leaves;
  Index _nmax;
};

}  // namespace Lpm
#endif
