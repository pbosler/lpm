#ifndef LPM_NODE_ARRAY_2D_INTERNAL_HPP
#define LPM_NODE_ARRAY_2D_INTERNAL_HPP

#include "LpmConfig.h"
#include "lpm_logger.hpp"
#include "tree/lpm_tree_defs.hpp"
#include "tree/lpm_tree_common.hpp"

namespace Lpm {
namespace quadtree {


struct NodeArray2DInternal {
  Kokkos::View<key_type*> node_keys;
  Kokkos::View<Index*> node_idx_start;
  Kokkos::View<id_type*> node_idx_count;
  Kokkos::View<id_type*> node_parents;
  Kokkos::View<id_type*[4]> node_kids;
  Int level;
  Int max_depth;
  id_type nnodes;

  NodeArray2DInternal() :
   level(NULL_IDX),
   max_depth(NULL_IDX),
   nnodes(0) {}

  template <typename LowerLevelType>
  NodeArray2DInternal(LowerLevelType& lower, const std::shared_ptr<Logger<>> mlog = nullptr) :
    level(lower.level-1),
    max_depth(lower.max_depth),
    nnodes(0),
    logger(mlog) {init(lower);}

  id_type max_pts_per_node() const;

  std::string info_string(const int tab_level=0) const;

  protected:
    void build_root(Kokkos::View<id_type*> level1_parents,
      const Kokkos::View<id_type*> level1_pt_count);

    template <typename LowerLevelType>
    void init(LowerLevelType& lower);

    std::shared_ptr<Logger<>> logger;
};

} // namespace quadtree
} // namespace Lpm

#endif
