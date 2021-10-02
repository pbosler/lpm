#include "tree/lpm_node_array_2d_internal.hpp"
#include "util/lpm_string_util.hpp"

namespace Lpm {
namespace quadtree {

void NodeArray2DInternal::build_root(Kokkos::View<id_type*> level1_parents,
  const Kokkos::View<id_type*> level1_pt_count) {
  nnodes = 1;
  node_keys = Kokkos::View<key_type*>("node_keys", 1);
  node_parents = Kokkos::View<id_type*>("node_parents", 1);
  node_kids = Kokkos::View<id_type*[4]>("node_kids", 1);
  node_idx_start = Kokkos::View<Index*>("node_idx_start", 1);
  node_idx_count = Kokkos::View<id_type*>("node_idx_count", 1);

  auto kids_local = node_kids;
  Kokkos::parallel_reduce(4, KOKKOS_LAMBDA (const id_type i, id_type& ct) {
    if (i==0) {
      node_keys(i) = 0;
      node_parents(i) = NULL_IDX;
      node_idx_start(i) = 0;
    }
    kids_local(0,i) = i;
    level1_parents(i) = 0;
    ct += level1_pt_count(i);
  }, node_idx_count(0));

#ifndef NDEBUG
  if (logger) {
    auto h_npts_view = Kokkos::create_mirror_view(node_idx_count);
    Kokkos::deep_copy(h_npts_view, node_idx_count);
    logger->debug("root node contains {} points.", h_npts_view());
  }
#endif
}

std::string NodeArray2DInternal::info_string(const int tab_level) const {
  std::ostringstream ss;
  auto tabstr = indent_string(tab_level);
  ss << tabstr << "NodeArray2DInternal info:\n";
  tabstr += "\t";
  ss << "nnodes = " << nnodes << "\n";
  ss << "level = " << level << "\n";
  ss << "max_depth = " << max_depth << "\n";
  ss << "max_pts_per_node = " << max_pts_per_node() << "\n";
  return ss.str();
}

id_type NodeArray2DInternal::max_pts_per_node() const {
  id_type result;
  auto local_pt_ct = node_idx_count;
  Kokkos::parallel_reduce(nnodes, KOKKOS_LAMBDA (const id_type i, id_type& ct) {
    if (local_pt_ct(i) > ct) ct = local_pt_ct(i);
  }, Kokkos::Max<id_type>(result));
  return result;
}

}
}
