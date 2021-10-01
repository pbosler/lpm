#ifndef LPM_TREE_NODE_HPP
#define LPM_TREE_NODE_HPP

#include "LpmConfig.h"
#include "tree/lpm_tree_defs.hpp"
#include "tree/lpm_tree_common.hpp"
#include "tree/lpm_box3d.hpp"
#include <memory>
#include <vector>

namespace Lpm {
namespace tree {

struct Node {
  Node(const Box3d& bb, Node* p = NULL,
    const std::vector<Index>& inds = std::vector<Index>()) :
    box(bb),
    level((p ? p->level+1 : 0)),
    parent(p),
    crd_inds(inds) {}

  virtual ~Node() {}

  inline Index n() const {return crd_inds.size();}
  inline bool is_leaf() const {return kids.empty();}
  inline bool has_kids() const {return !is_leaf();}
  inline bool empty() const {return crd_inds.empty();}

  Box3d box;
  int level;
  Node* parent;
  std::vector<std::unique_ptr<Node>> kids;
  std::vector<Index> crd_inds;

  std::string info_string(const int tab_level = 0) const;

};

} // namespace tree
} // namespace Lpm

#endif
