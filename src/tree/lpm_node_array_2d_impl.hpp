#ifndef LPM_NODE_ARRAY_2D_IMPL_HPP
#define LPM_NODE_ARRAY_2D_IMPL_HPP

#include "LpmConfig.h"
#include "tree/lpm_node_array_2d.hpp"

namespace Lpm {
namespace quadtree {

struct EncodeFunctor {
  // output
  Kokkos::View<code_type*> codes;
  // input
  Kokkos::View<Real*[2]> pts;
  Int level;
  Box2d bounding_box;

  /** @brief constructor.

    @param [in/out] c view to store output sort codes
    @param [in] p points
    @param [in] l level
  */
  EncodeFunctor(Kokkos::View<code_type*> c, const Kokkos::View<Real*[2]> p,
    const Int l, const Box2d bb) :
    codes(c),
    pts(p),
    level(l),
    bounding_box(bb) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const id_type i) const {
    const auto pos = Kokkos::subview(pts, i, Kokkos::ALL);
    const auto key = compute_key_for_point(pos, level, bounding_box);
    codes(i) = encode(key, i);
  }
};

struct SortPointsFunctor {
  // output
  Kokkos::View<Real*[2]> sorted_pts;
  Kokkos::View<Index*> orig_inds;
  // input
  Kokkos::View<Real*[2]> unsorted_pts;
  Kokkos::View<code_type*> sort_codes;

  SortPointsFunctor(Kokkos::View<Real*[2]> sp,
    Kokkos::View<Index*> inds,
    const Kokkos::View<Real*[2]> up,
    const Kokkos::View<code_type*> c) : sorted_pts(sp),
                                        orig_inds(inds),
                                        unsorted_pts(up),
                                        sort_codes(c) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const id_type i) const {
    const auto old_id = decode_id(sort_codes(i));
    orig_inds(i) = old_id;
    for (auto j=0; j<2; ++j) {
      sorted_pts(i,j) = unsorted_pts(old_id, j);
    }
  }
};

struct UniqueKeyFunctor {
  Kokkos::View<id_type*> unique_flags;
  Kokkos::View<code_type*> sort_codes;

  UniqueKeyFunctor(Kokkos::View<id_type*> uf, Kokkos::View<code_type*> sc) :
    unique_flags(uf),
    sort_codes(sc) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const id_type i, id_type& ct) const {
    if (i == 0) {
      unique_flags(i) = 1;
    }
    else {
      const auto prev_key = decode_key(sort_codes(i-1));
      const auto key = decode_key(sort_codes(i));
      unique_flags(i) = (key != prev_key);
    }
    ct += unique_flags(i);
  }
};

struct UniqueNodeFunctor {
  Kokkos::View<key_type*> keys;
  Kokkos::View<Index*> pt_idx_start;
  Kokkos::View<id_type*> pt_idx_count;
  Kokkos::View<id_type*> scanned_flags;
  Kokkos::View<code_type*> sorted_codes;

  UniqueNodeFunctor(Kokkos::View<key_type*> k,
                    Kokkos::View<Index*> idx0,
                    Kokkos::View<id_type*> idxn,
                    Kokkos::View<id_type*> f,
                    Kokkos::View<code_type*> c):
    keys(k),
    pt_idx_start(idx0),
    pt_idx_count(idxn),
    scanned_flags(f),
    sorted_codes(c) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const id_type i) const {
    bool is_new_key = true;
    if (i>0) is_new_key = (scanned_flags(i) > scanned_flags(i-1));
    if (is_new_key) {
      const auto new_idx = scanned_flags(i) - 1;
      const auto new_key = decode_key(sorted_codes(i));
      const auto first_pt = binary_search_first(new_key, sorted_codes);
      const auto last_pt = binary_search_last(new_key, sorted_codes);
      keys(new_idx) = new_key;
      pt_idx_start(new_idx) = first_pt;
      pt_idx_count(new_idx) = last_pt - first_pt + 1;
      LPM_KERNEL_ASSERT(pt_idx_count(new_idx) > 0);
    }
  }
};

struct NodeNumFunctor {
  Kokkos::View<id_type*> node_num;
  Kokkos::View<key_type*> unique_keys;
  Int level;
  Int max_depth;

  NodeNumFunctor(Kokkos::View<id_type*> num,
      const Kokkos::View<key_type*> ukeys,
      const Int l, const Int md) :
      node_num(num), unique_keys(ukeys), level(l), max_depth(md) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const id_type i) const {
    if (i==0) {
      node_num(i) = 4;
    }
    else {
      const auto prev_parent = parent_key(unique_keys(i-1), level, max_depth);
      const auto parent = parent_key(unique_keys(i), level, max_depth);
      node_num(i) = (parent == prev_parent ? 0 : 4);
    }
  }
};

struct NodeArray2DFunctor {
  Kokkos::View<key_type*> keys;
  Kokkos::View<Index*> pt_idx_start;
  Kokkos::View<id_type*> pt_idx_count;
  Kokkos::View<id_type*> parent_idx;
  Kokkos::View<id_type*> pt_node;
  Kokkos::View<id_type*> node_num;
  Kokkos::View<key_type*> unique_keys;
  Kokkos::View<Index*> unique_idx_start;
  Kokkos::View<id_type*> unique_idx_count;
  Int level;

  NodeArray2DFunctor(Kokkos::View<key_type*> k, Kokkos::View<Index*> idx0,
    Kokkos::View<id_type*> idxn, Kokkos::View<id_type*> pidx,
    Kokkos::View<id_type*> ptn, const Kokkos::View<id_type*> num,
    const Kokkos::View<key_type*> ukeys, const Kokkos::View<Index*> uidx0,
    const Kokkos::View<id_type*> uidxn, const Int l) :
    keys(k),
    pt_idx_start(idx0),
    pt_idx_count(idxn),
    parent_idx(pidx),
    pt_node(ptn),
    node_num(num),
    unique_keys(ukeys),
    unique_idx_start(uidx0),
    unique_idx_count(uidxn),
    level(l) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const id_type i) const {
    bool is_new_parent = true;
    if (i>0) is_new_parent = (node_num(i) > node_num(i-1));
    if (is_new_parent) {
      const auto kid0_idx = node_num(i) - 4;
      const auto p_key = parent_key(unique_keys(i), level, level);
      for (int k=0; k<4; ++k) {
        const id_type k_idx = kid0_idx + k;
        const key_type k_key = p_key + k;
        keys(k_idx) = k_key;
        parent_idx(k_idx) = NULL_IDX;
        const auto ukey_idx = binary_search_first(k_key, unique_keys);
        if (ukey_idx != NULL_IDX) {
          pt_idx_start(k_idx) = unique_idx_start(ukey_idx);
          pt_idx_count(k_idx) = unique_idx_count(ukey_idx);
          for (auto j=0; j<unique_idx_count(ukey_idx); ++j) {
            pt_node(unique_idx_start(ukey_idx) + j) = k_idx;
          }
        }
        else {
          pt_idx_start(k_idx) = NULL_IDX;
          pt_idx_count(k_idx) = 0;
        }
      }
    }
  }
};

} // namespace quadtree
} // namespace Lpm

#endif
