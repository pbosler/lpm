#ifndef LPM_NODE_ARRAY_2D_HPP
#define LPM_NODE_ARRAY_2D_HPP

#include "LpmConfig.h"
#include "tree/lpm_tree_defs.hpp"
#include "tree/lpm_tree_common.hpp"
#include "tree/lpm_quadtree_functions.hpp"
#include "lpm_logger.hpp"
#include <memory>

namespace Lpm {
namespace quadtree {

struct NodeArray2D {
  Kokkos::View<Real*[2]> sorted_pts;
  Int level;
  Int max_depth;
  id_type nnodes;
  id_type npts;

  Kokkos::View<id_type*> pt_node;
  Kokkos::View<Index*> pt_idx_orig;
  Kokkos::View<Box2d> bounding_box;

  Kokkos::View<key_type*> node_keys;
  Kokkos::View<Index*> node_idx_start;
  Kokkos::View<id_type*> node_idx_count;
  Kokkos::View<id_type*> node_parents;

  NodeArray2D(const Kokkos::View<Real*[2]> unsorted_pts, const Int d,
    const std::shared_ptr<Logger<>> mlog = nullptr) :
    sorted_pts("sorted_pts", unsorted_pts.extent(0)),
    level(d),
    max_depth(d),
    nnodes(0),
    npts(unsorted_pts.extent(0)),
    pt_node("pt_node", unsorted_pts.extent(0)),
    pt_idx_orig("pt_idx_orig", unsorted_pts.extent(0)),
    bounding_box("bounding_box"),
    logger(mlog) {
      init(unsorted_pts);
    }

  void write_vtk(const std::string& ofname) const;

  std::string info_string(const int tab_level=0) const;

  id_type max_pts_per_node() const;

  protected:
    void init(const Kokkos::View<Real*[2]> unsorted_pts);

    std::shared_ptr<Logger<>> logger;
};

} // namespace quadtree
} // namespace Lpm

#endif
