#ifndef LPM_QUADTREE_FUNCTIONS_HPP
#define LPM_QUADTREE_FUNCTIONS_HPP

#include "LpmConfig.h"
#include "lpm_kokkos_defs.hpp"
#include "tree/lpm_tree_defs.hpp"
#include "tree/lpm_tree_common.hpp"
#include "tree/lpm_box2d.hpp"

namespace Lpm {
namespace quadtree {

template <typename PointType> KOKKOS_INLINE_FUNCTION
key_type compute_key_for_point(const PointType& pos, const int depth,
  const Box2d& bounding_box=default_box()) {
  LPM_KERNEL_ASSERT(depth >=0 && depth <= MAX_QUADTREE_DEPTH);
  LPM_KERNEL_REQUIRE(bounding_box.is_square());
  auto c = bounding_box.centroid();
  auto half_len = 0.5*bounding_box.square_edge_length();

  const auto nbits = 2*depth; // key length in bits
  key_type key = 0;
  for (auto i=1; i<=depth; ++i) {
    half_len *= 0.5;
    const auto yb = nbits - 2*i; // position of level i's y-bit
    const bool righty = pos(1) > c[1];
    const bool rightx = pos(0) > c[0];

    key += (righty ? pow2<key_type>(yb) : 0);
    c[1] += (righty ? half_len : -half_len);
    key += (rightx ? pow2<key_type>(yb+1) : 0);
    c[0] += (rightx ? half_len : -half_len);
  }
  return key;
}

KOKKOS_INLINE_FUNCTION
key_type parent_key(const key_type& kid_key, const int lev, const int max_depth) {
  LPM_KERNEL_ASSERT( max_depth > 0 && max_depth <= MAX_QUADTREE_DEPTH );
  LPM_KERNEL_ASSERT( lev > 0 && lev <= max_depth );

  const auto nbits = 2*max_depth;
  const auto pyb = nbits - 2*(lev-1);
  key_type mask = 0;
  for (auto i=nbits; i>=pyb; --i) {
    mask += pow2<key_type>(i);
  }
  return (kid_key & mask);
}

KOKKOS_INLINE_FUNCTION
key_type local_key(const key_type& key, const int lev, const int max_depth) {
  LPM_KERNEL_ASSERT( max_depth > 0 && max_depth <= MAX_QUADTREE_DEPTH );
  LPM_KERNEL_ASSERT( lev > 0 && lev <= max_depth );

  const auto nbits = 2*max_depth;
  const auto pyb = nbits - 2*(lev-1);
  key_type mask = 0;
  for (auto i=pyb-2; i<pyb; ++i) {
    mask += pow2<key_type>(i);
  }
  return ( (key & mask) >> (pyb-2) );
}

KOKKOS_INLINE_FUNCTION
Box2d box_from_key(const key_type& k, const int lev, const int max_depth,
  const Box2d& bounding_box=default_box()) {
  LPM_KERNEL_ASSERT(max_depth > 0 && max_depth <= MAX_QUADTREE_DEPTH);
  LPM_KERNEL_ASSERT( lev >= 0 && lev <= max_depth);
  auto c = bounding_box.centroid();
  auto half_len = 0.5 * bounding_box.square_edge_length();
  for (auto i=1; i<=lev; ++i) {
    half_len *= 0.5;
    const key_type lkey = local_key(k, i, max_depth);
    c[1] += ( (lkey&1) ? half_len : -half_len);
    c[0] += ( (lkey&2) ? half_len : -half_len);
  }
  return Box2d(c[0] - half_len, c[0] + half_len,
               c[1] - half_len, c[1] + half_len, false);
}

KOKKOS_INLINE_FUNCTION
key_type build_key(const key_type& parent_key, const int kid_idx,
  const int kid_lev, const int max_depth) {
  LPM_KERNEL_ASSERT(max_depth > 0 and max_depth <= MAX_QUADTREE_DEPTH);
  LPM_KERNEL_ASSERT(kid_lev > 0 and kid_lev <= max_depth);
  const auto pzb = 2*max_depth - 2*(kid_lev-1);
  const key_type shifted_kid = (kid_idx << (pzb-2));
  return parent_key + shifted_kid;
}

} // namespace quadtree
} // namespace Lpm

#endif
