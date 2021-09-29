#ifndef LPM_NODE_ARRAY_D_HPP
#define LPM_NODE_ARRAY_D_HPP

#include "LpmConfig.h"
#include "tree/lpm_gpu_octree_functions.hpp"

namespace Lpm {
namespace tree {

/** @brief Constructs the leaves of the octree in parallel

    Implements Listing 1 from:

    K. Zhou, et. al., 2011. Data-parallel octrees for surface reconstruction,
    IEEE Trans. Vis. Comput. Graphics 17(5): 669--681. DOI: 10.1109/TVCG.2010.75
*/
struct NodeArrayD {
  Kokkos::View<Real*[3]> sorted_pts;
  Int level; // depth of these leaves
  Int max_depth; // maximum depth of octree
  id_type nnodes; // number of leaf nodes

  // allocated at initialization
  Kokkos::View<id_type*> pt_in_node; // pt_in_node(i) = the idx of the node that contains pt(i)
  Kokkos::View<Index*> pt_idx_orig; // pt_idx_orig(i) = the original (unsorted) idx of sorted pt(i)
  Kokkos::View<Box3d> bounding_box; // bounding box for the whole point set

  // allocated inside the init function
  /// node_keys(t) = shuffled xyz key of node t
  Kokkos::View<key_type*> node_keys;
  /// node_idx_start(t) = starting point idx in sorted_pts for points contained by node t
  Kokkos::View<Index*> node_idx_start;
  /// node_idx_start(t) = the number of points contained by node t
  Kokkos::View<id_type*> node_idx_count;
  /// node_parents(t) = the idx of node(t)'s parent.
  /// allocated here in the init method, but set by the octree level above this one.
  Kokkos::View<id_type*> node_parents;

  /** @brief constructor.

    @param [in] usp unsorted point array
    @param [in] d maximum depth of octree
  */
  NodeArrayD(const Kokkos::View<Real*[3]> usp, const Int d) :
    sorted_pts("sorted_pts", usp.extent(0)),
    level(d),
    max_depth(d),
    pt_in_node("pt_in_node", usp.extent(0)),
    pt_idx_orig("pt_idx_orig", usp.extent(0)),
    bounding_box("bounding_box") {
      init(usp);
    }

  void init(const Kokkos::View<Real*[3]> unsorted_pts);

  void write_vtk(const std::string& ofname) const;

  std::string info_string(const int tab_level=0) const;

  typename Kokkos::MinMax<id_type>::value_type min_max_pts_per_node() const;
};

} // namespace tree
} // namespace Lpm

#endif
