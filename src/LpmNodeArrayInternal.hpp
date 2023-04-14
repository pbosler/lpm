#ifndef LPM_NODE_ARRAY_INTERNAL_HPP
#define LPM_NODE_ARRAY_INTERNAL_HPP

#include <string>

#include "Kokkos_Core.hpp"
#include "LpmBox3d.hpp"
#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmKokkosUtil.hpp"
#include "LpmNodeArrayD.hpp"
#include "LpmOctreeUtil.hpp"

namespace Lpm {
namespace Octree {

/**
    Class must initialize its own arrays
        * and *
    Set the lower level's node_parents values.

    These actions are performed in the constructors.
*/
class NodeArrayInternal {
 public:
  Int level;
  Int max_depth;

  ko::View<key_type*> node_keys;  // keys of nodes at this level
  ko::View<Index* [2]> node_pt_inds;

  ko::View<Index*> node_parents;   // address of parents of nodes at this level
                                   // (into level-1 NodeArrayInternal)
  ko::View<Index* [8]> node_kids;  // address of children (in level+1 NodeArray)
  ko::View<BBox> root_box;

  NodeArrayInternal() {}

  NodeArrayInternal(NodeArrayD& leaves)
      : level(leaves.depth - 1), max_depth(leaves.depth), root_box(leaves.box) {
    initFromLeaves(leaves);
  }

  NodeArrayInternal(NodeArrayInternal& lower)
      : level(lower.level - 1),
        max_depth(lower.max_depth),
        root_box(lower.root_box) {
    initFromLower(lower);
  }

  std::string infoString(const bool& verbose = false) const;

  void initFromLeaves(NodeArrayD& leaves);

  void initFromLower(NodeArrayInternal& lower);

 protected:
};

}  // namespace Octree
}  // namespace Lpm
#endif