#ifndef LPM_NODE_ARRAY_2D_INTERNAL_IMPL_HPP
#define LPM_NODE_ARRAY_2D_INTERNAL_IMPL_HPP

#include "tree/lpm_node_array_2d_internal.hpp"
#include "tree/lpm_node_array_2d_impl.hpp"
#include "tree/lpm_quadtree_functions.hpp"
#include "lpm_assert.hpp"

namespace Lpm {
namespace quadtree {

struct ParentsFunctor {
  // output
  Kokkos::View<key_type*> parent_keys;
  Kokkos::View<Index*> parent_idx_start;
  Kokkos::View<id_type*> parent_idx_count;
  // input
  Kokkos::View<key_type*> keys_lower;
  Kokkos::View<Index*> lower_idx_start;
  Kokkos::View<id_type*> lower_idx_count;
  Int level;
  Int max_depth;

  ParentsFunctor(Kokkos::View<key_type*> pkeys,
                 Kokkos::View<Index*> pidx0,
                 Kokkos::View<id_type*> pidxn,
           const Kokkos::View<key_type*> lkeys,
           const Kokkos::View<Index*> lidx0,
           const Kokkos::View<id_type*> lidxn,
           const Int l,
           const Int md) :
    parent_keys(pkeys),
    parent_idx_start(pidx0),
    parent_idx_count(pidxn),
    keys_lower(lkeys),
    lower_idx_start(lidx0),
    lower_idx_count(lidxn),
    level(l),
    max_depth(md) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const id_type i) const {
    const auto kid0_address = 4*i;
    const auto my_key = parent_key(keys_lower(kid0_address), level+1, max_depth);
    parent_keys(i) = my_key;
    parent_idx_count(i) = 0;
    for (auto k=0; k<4; ++k) {
      const auto kid_idx = kid0_address + k;
      bool found_first = false;
      if (lower_idx_count(kid_idx) > 0 and !found_first) {
        parent_idx_start(kid_idx) = lower_idx_start(kid_idx);
        found_first = true;
      }
      parent_idx_count(i) += lower_idx_count(kid_idx);
    }
    LPM_KERNEL_ASSERT(parent_idx_count(i) > 0);
  }
};

struct NodeArray2DInternalFunctor {
  // output
  Kokkos::View<key_type*> node_keys;
  Kokkos::View<Index*> node_idx_start;
  Kokkos::View<id_type*> node_idx_count;
  Kokkos::View<id_type*> node_parents;
  Kokkos::View<id_type*[4]> node_kids;
  Kokkos::View<id_type*> parents_lower;
  // input
  Kokkos::View<id_type*> node_num;
  Kokkos::View<key_type*> parent_keys;
  Kokkos::View<Index*> parent_idx_start;
  Kokkos::View<id_type*> parent_idx_count;
  Kokkos::View<key_type*> keys_lower;
  Int level;
  Int max_depth;

  NodeArray2DInternalFunctor(Kokkos::View<key_type*> keys,
                             Kokkos::View<Index*> idx0,
                             Kokkos::View<id_type*> idxn,
                             Kokkos::View<id_type*> np,
                             Kokkos::View<id_type*[4]> nk,
                             Kokkos::View<id_type*> lp,
                       const Kokkos::View<id_type*> nn,
                       const Kokkos::View<key_type*> pkeys,
                       const Kokkos::View<Index*> pidx0,
                       const Kokkos::View<id_type*> pidxn,
                       const Kokkos::View<key_type*> lkeys,
                       const Int l,
                       const Int md) :
    node_keys(keys),
    node_idx_start(idx0),
    node_idx_count(idxn),
    node_parents(np),
    node_kids(nk),
    parents_lower(lp),
    node_num(nn),
    parent_keys(pkeys),
    parent_idx_start(pidx0),
    parent_idx_count(pidxn),
    keys_lower(lkeys),
    level(l),
    max_depth(md) {}


  KOKKOS_INLINE_FUNCTION
  void operator() (const id_type i) const {
    bool new_grandparent = true;
    if (i>0) new_grandparent = (node_num(i) > node_num(i-1));
    if (new_grandparent) {
      const auto grandparent_key = parent_key(parent_keys(i), level, max_depth);
      const auto parent0_address = node_num(i)-4;

      for (auto p=0; p<4; ++p) {
        const auto parent_idx = parent0_address + p;
        const auto pkey = build_key(grandparent_key, p, level, max_depth);
        node_keys(parent_idx) = pkey;
        node_parents(parent_idx) = NULL_IDX;
        const auto key_idx = binary_search_first(pkey, parent_keys);
        const bool key_found = (key_idx != NULL_IDX);
        if (key_found) {
          node_idx_start(parent_idx) = parent_idx_start(key_idx);
          node_idx_count(parent_idx) = parent_idx_count(key_idx);
          const auto kid0_idx = binary_search_first(pkey, keys_lower);
          LPM_KERNEL_ASSERT(kid0_idx != NULL_IDX);
          for (auto k=0; k<4; ++k) {
            node_kids(parent_idx,k) = kid0_idx + k;
            parents_lower(kid0_idx+k) = parent_idx;
          }
        }
        else {
          node_idx_start(parent_idx) = NULL_IDX;
          node_idx_count(parent_idx) = 0;
          for (auto k=0; k<4; ++k) {
            node_kids(parent_idx,k) = NULL_IDX;
          }
        }
      }
    }
  }
};

template <typename LowerLevelType>
void NodeArray2DInternal::init(LowerLevelType& lower) {
  LPM_ASSERT(lower.nnodes%4 == 0);
  if (level == 0) {
    build_root(lower.node_parents, lower.node_idx_count);
  }
  else {
    const auto nparents = lower.nnodes/4;
    Kokkos::View<key_type*> parent_keys("parent_keys", nparents);
    Kokkos::View<Index*> parent_idx0("parent_idx0", nparents);
    Kokkos::View<id_type*> parent_idxn("parent_idxn", nparents);

    Kokkos::parallel_for(nparents, ParentsFunctor(parent_keys, parent_idx0,
      parent_idxn, lower.node_keys, lower.node_idx_start, lower.node_idx_count,
      level, max_depth));

#ifndef NDEBUG
  if (logger) {
    for (auto i=0; i<nparents; ++i) {
      logger->trace("{}: parent key {} idx0 {} idxn {}", i, parent_keys(i),
        parent_idx0(i), parent_idxn(i));
    }
  }
#endif


    Kokkos::View<id_type*> node_num("node_num", nparents);
    Kokkos::parallel_for(nparents, NodeNumFunctor(node_num, parent_keys,
      level, max_depth));

    Kokkos::parallel_scan(nparents,
      KOKKOS_LAMBDA (const id_type i, id_type& update, const bool is_final) {
        const auto num_i = node_num(i);
        update += num_i;
        if (is_final) {
          node_num(i) = update;
        }
      });

    const auto nnodes_view = Kokkos::subview(node_num, nparents-1);
    auto h_nnodes_view = Kokkos::create_mirror_view(nnodes_view);
    Kokkos::deep_copy(h_nnodes_view, nnodes_view);
    nnodes = h_nnodes_view();

    LPM_ASSERT(nnodes >= nparents);
#ifndef NDEBUG
  if (logger) {
    logger->debug("level {} will have {} nodes (including empty siblings).",
      level, nnodes);
  }
#endif

    node_keys = Kokkos::View<key_type*>("node_keys", nnodes);
    node_idx_start = Kokkos::View<Index*>("node_idx_start", nnodes);
    node_idx_count = Kokkos::View<id_type*>("node_idx_count", nnodes);
    node_parents = Kokkos::View<id_type*>("node_parents", nnodes);
    node_kids = Kokkos::View<id_type*[4]>("node_kids", nnodes);

    Kokkos::parallel_for(nparents, NodeArray2DInternalFunctor(node_keys,
      node_idx_start, node_idx_count, node_parents, node_kids,
      lower.node_parents, node_num, parent_keys, parent_idx0, parent_idxn,
      lower.node_keys, level, max_depth));
  }
}

} // namespace quadtree
} // namespace Lpm

#endif
