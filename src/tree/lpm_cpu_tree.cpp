#include "tree/lpm_cpu_tree.hpp"
#include "lpm_assert.hpp"
#include <numeric>
#include <limits>

namespace Lpm {
namespace tree {

template <typename Geo>
CpuTree<Geo>::CpuTree(const std::shared_ptr<Coords<Geo>> crds) :
  _crds(crds),
  depth(0),
  n_nodes(1) {

  const auto xminmax = crds->min_max_extent(0);
  const auto yminmax = crds->min_max_extent(1);
  const auto zminmax = crds->min_max_extent(2);

  Box3d root_box(xminmax.min_val, xminmax.max_val,
                 yminmax.min_val, yminmax.max_val,
                 zminmax.min_val, zminmax.max_val);

  std::vector<Index> root_inds(crds->nh());
  std::iota(root_inds.begin(), root_inds.end(), 0);

  root = std::unique_ptr<Node>(new Node(root_box, NULL, root_inds));

  LPM_ASSERT(root->is_leaf());
  LPM_ASSERT(!root->has_kids());
  LPM_ASSERT(root->n() == crds.size());
}

template <typename Geo>
void CpuTree<Geo>::shrink_box(Node* node) {
  Real xmin = std::numeric_limits<Real>::max();
  Real xmax = std::numeric_limits<Real>::lowest();
  Real ymin = xmin;
  Real ymax = xmax;
  Real zmin = xmin;
  Real zmax = xmax;
  const auto crd_view = _crds->get_host_crd_view();
  for (Index i=0; i<node->n(); ++i) {
    const auto cxyz = Kokkos::subview(crd_view, i, Kokkos::ALL);
    if (cxyz[0] < xmin) xmin = cxyz[0];
    if (cxyz[1] < ymin) ymin = cxyz[1];
    if (cxyz[2] < zmin) zmin = cxyz[2];
    if (cxyz[0] > xmax) xmax = cxyz[0];
    if (cxyz[1] > ymax) ymax = cxyz[1];
    if (cxyz[2] > zmax) zmax = cxyz[2];
  }
  node->box.xmin = xmin;
  node->box.xmax = xmax;
  node->box.ymin = ymin;
  node->box.ymax = ymax;
  node->box.zmin = zmin;
  node->box.zmax = zmax;
}

template <typename Geo>
void CpuTree<Geo>::divide_node(Node* node, const bool do_shrink) {
  LPM_ASSERT(!node->empty());
  const auto kid_boxes = node->box.bisect_all();
  auto counted = std::vector<bool>(node->n(), false);
  const auto crd_view = _crds->get_host_crd_view();
  int empty_count = 0;
  int full_count = 0;
  for (int k=0; k<8; ++k) {
    std::vector<Index> kid_inds;
    for (int i=0; i<node->n(); ++i) {
      const auto cxyz = Kokkos::subview(crd_view, i, Kokkos::ALL);
      if (kid_boxes[k].contains_pt(cxyz)) {
        if (!counted[i]) {
          kid_inds.push_back(i);
          counted[i] = true;
        }
      }
    }
    if (kid_inds.empty()) {
      ++empty_count;
    }
    else {
      ++full_count;
      kid_inds.shrink_to_fit();
    }
    node->kids.push_back(std::unique_ptr<Node>(
      new Node(kid_boxes[k], node, kid_inds)));
  }
  LPM_ASSERT(node->has_kids());
  LPM_ASSERT(full_count + empty_count == 8);
  LPM_ASSERT(full_count > 0);
  n_nodes += full_count;
  if (do_shrink) {
    for (int k=0; k<8; ++k) {
      if (!node->kids[k]->empty()) {
        shrink_box(node->kids[k].get());
      }
    }
  }
}

template <typename Geo>
void CpuTree<Geo>::generate_tree_max_coords_per_node(Node* node,
  const Index max_coords_per_node, const bool do_shrink) {
  if (node->n() <= max_coords_per_node or node->empty()) {
    return;
  }
  else {
    divide_node(node, do_shrink);
    for (int k=0; k<8; ++k) {
      generate_tree_max_coords_per_node(node->kids[k].get(),
        max_coords_per_node, do_shrink);
    }
  }
}

template <typename Geo>
void CpuTree<Geo>::generate_tree_max_depth(Node* node, const int max_depth,
  const bool do_shrink) {
  if (node->level == max_depth or node->empty()) {
    return;
  }
  else {
    divide_node(node, do_shrink);
    for (int k=0; k<8; ++k) {
      generate_tree_max_depth(node->kids[k].get(), max_depth, do_shrink);
    }
  }
}



} // namespace tree
} // namespace Lpm
