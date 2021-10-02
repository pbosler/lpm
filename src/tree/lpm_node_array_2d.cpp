#include "tree/lpm_node_array_2d.hpp"
#include "tree/lpm_node_array_2d_impl.hpp"
#include "lpm_box2d.hpp"
#include "util/lpm_string_util.hpp"
#include "Kokkos_Sort.hpp"

namespace Lpm {
namespace quadtree {

void NodeArray2D::init(const Kokkos::View<Real*[2]> unsorted_pts) {

  Kokkos::parallel_reduce(npts, BoundingBoxFunctor(unsorted_pts),
    Box2dReducer<>(bounding_box()));

  auto h_bounding_box = Kokkos::create_mirror_view(bounding_box);
      Kokkos::deep_copy(h_bounding_box, bounding_box);

  h_bounding_box().make_square();
  bbox = h_bounding_box();
  Kokkos::deep_copy(bounding_box, h_bounding_box);


#ifndef NDEBUG
    if (logger) {
      logger->debug("NodeArray2D bouding box: {}", h_bounding_box());
    }
#endif

  Kokkos::View<code_type*> sort_codes("sort_codes", npts);
  Kokkos::parallel_for(npts, EncodeFunctor(sort_codes, unsorted_pts, level, bbox));

  Kokkos::sort(sort_codes);
  Kokkos::parallel_for(npts, SortPointsFunctor(sorted_pts, pt_idx_orig,
    unsorted_pts, sort_codes));

  Kokkos::View<id_type*> unique_flags("unique_flags", npts);
  id_type u_count;
  Kokkos::parallel_reduce(npts, UniqueKeyFunctor(unique_flags, sort_codes),
    u_count);

#ifndef NDEBUG
  if (logger) {
     logger->debug("found {} unique keys in {} points.", u_count, npts);
  }
#endif

  Kokkos::parallel_scan("scan unique keys", npts,
    KOKKOS_LAMBDA (const id_type i, id_type& update, const bool is_final) {
      const auto flag_i = unique_flags(i);
      update += flag_i;
      if (is_final) {
        unique_flags(i) = update;
      }
    });

  Kokkos::View<key_type*> unique_keys("unique_keys", u_count);
  Kokkos::View<Index*> upt_idx_start("upt_idx_start", u_count);
  Kokkos::View<id_type*> upt_idx_count("upt_idx_count", u_count);
  Kokkos::parallel_for(npts, UniqueNodeFunctor(unique_keys, upt_idx_start,
    upt_idx_count, unique_flags, sort_codes));

#ifndef NDEBUG
  if (logger) {
    id_type pt_ct;
    Kokkos::parallel_reduce(u_count, KOKKOS_LAMBDA (const Index i, id_type& ct) {
      if (upt_idx_count(i) > ct) ct = upt_idx_count(i);
    }, Kokkos::Max<id_type>(pt_ct));
    logger->debug("max pts per node: {} out of {} total", pt_ct, npts);
  }
#endif

  Kokkos::View<id_type*> node_num("node_num", u_count);
  Kokkos::parallel_for(u_count, NodeNumFunctor(node_num, unique_keys, level, max_depth));

  Kokkos::parallel_scan("scan unique parents, add siblings", u_count,
    KOKKOS_LAMBDA (const id_type i, id_type& update, const bool is_final) {
      const auto num_i = node_num(i);
      update += num_i;
      if (is_final) {
        node_num(i) = update;
      }
    });

  const auto nnodes_view = Kokkos::subview(node_num, u_count-1);
  auto h_nnodes_view = Kokkos::create_mirror_view(nnodes_view);
  Kokkos::deep_copy(h_nnodes_view, nnodes_view);
  nnodes = h_nnodes_view();

  LPM_ASSERT(nnodes >= u_count);

#ifndef NDEBUG
  if (logger) {
    logger->debug("including empty siblings makes {} leaves total at level {}.",
     nnodes, level);
  }
#endif

  node_keys = Kokkos::View<key_type*>("node_keys", nnodes);
  node_idx_start = Kokkos::View<Index*>("node_idx_start", nnodes);
  node_idx_count = Kokkos::View<id_type*>("node_idx_count", nnodes);
  node_parents = Kokkos::View<id_type*>("node_parents", nnodes);

  Kokkos::parallel_for(u_count, NodeArray2DFunctor(node_keys, node_idx_start,
    node_idx_count, node_parents, pt_node, node_num, unique_keys,
    upt_idx_start, upt_idx_count, level));

}

std::string NodeArray2D::info_string(const int tab_level) const {
  std::ostringstream ss;
  auto tabstr = indent_string(tab_level);
  ss << tabstr << "NodeArray2D info:\n";
  tabstr += "\t";
  ss << tabstr << "nnodes = " << nnodes << "\n";
  ss << tabstr << "level = " << level << "\n";
  ss << tabstr << "max_depth = " << max_depth << "\n";
  ss << tabstr << "max_pts_per_node = " << max_pts_per_node() << "\n";
  return ss.str();
}

id_type NodeArray2D::max_pts_per_node() const {
  id_type result;
  const auto local_pt_ct_view = node_idx_count;
  Kokkos::parallel_reduce(nnodes, KOKKOS_LAMBDA (const Index i, id_type& ct) {
    if (local_pt_ct_view(i) > ct) ct = local_pt_ct_view(i);
  }, Kokkos::Max<id_type>(result));
  return result;
}

}
}
