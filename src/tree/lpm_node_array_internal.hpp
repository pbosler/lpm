#ifndef LPM_NODE_ARRAY_INTERNAL_HPP
#define LPM_NODE_ARRAY_INTERNAL_HPP

#include "LpmConfig.h"
#include "tree/lpm_tree_defs.hpp"
#include "tree/lpm_tree_common.hpp"
#include "tree/lpm_gpu_octree_functions.hpp"
#include "tree/lpm_node_array_d.hpp"

namespace Lpm {
namespace octree {

/** @brief Constructs internal levels of the octree in parallel

    K. Zhou, et. al., 2011. Data-parallel octrees for surface reconstruction,
    IEEE Trans. Vis. Comput. Graphics 17(5): 669--681. DOI: 10.1109/TVCG.2010.75
*/
struct NodeArrayInternal {
  /// level of this set of internal nodes
  Int level;
  /// max depth of tree (depth of leaves)
  Int max_depth;
  /// number of nodes in this level
  id_type nnodes;

  /// shuffled xyz keys of nodes at this level
  Kokkos::View<key_type*> node_keys;
  /// idx to sorted points array of the first point contained by each node at this level
  Kokkos::View<Index*> node_idx_start;
  /// number of points contained by each node
  Kokkos::View<id_type*> node_idx_count;
  /// parents of nodes at this level
  /// Note: parents are allocated here, but not initialized.
  /// Parents are initialized by the next level up in the octree.
  Kokkos::View<id_type*> node_parents;
  /// kids of nodes at this level
  Kokkos::View<id_type*[8]> node_kids;

  /// Constructor
  template <typename LowerLevelType>
  NodeArrayInternal(LowerLevelType& lower) :
    level(lower.level-1),
    max_depth(lower.max_depth),
    nnodes(0) { init_from_lower(lower); }

  std::string info_string(const int tab_level=0) const;

  void write_vtk(const std::string& ofname) const;

  typename Kokkos::MinMax<id_type>::value_type min_max_pts_per_node() const;

  protected:
    void build_root(Kokkos::View<id_type*> parents_lower,
              const Kokkos::View<id_type*> idx_count_lower);

    template <typename LowerLevelType>
    void init_from_lower(LowerLevelType& lower);
};

} // namespace tree
} // namespace Lpm

#endif
