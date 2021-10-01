#ifndef LPM_GPU_OCTREE_FUNCTIONS_HPP
#define LPM_GPU_OCTREE_FUNCTIONS_HPP

#include "LpmConfig.h"
#include "lpm_assert.hpp"
#include "lpm_constants.hpp"
#include "tree/lpm_tree_defs.hpp"
#include "tree/lpm_tree_common.hpp"
#include "tree/lpm_box3d.hpp"
#include <array>

/**
  This file contains functions and utilities for use with the data-parallel
  octrees defined in

  Zhou, Gong, Huang, Guo, 2011, Data-parallel octrees for surface reconstruction,
  IEEE Trans. Vis. Comput. Graphics, 17(5) 669 -- 681.

*/
namespace Lpm {
namespace octree {


/** @brief Compute the shuffled xyz key associated with a point at pos,

  key = x1y1z1x2y2z2...xdydzd

  where bits xi, yi, zi correspond to left (0) or right(1) of the centroid
  in that coordinate direction of the octree node at level i.

  @param [in] pos xyz coordinates of point
  @param [in] depth depth of node in octree where the key is required
  @param [in] bounding_box bounding box of the entire point set (the box of the
    octree's root node)
  @return shuffled xyz key
*/
template <typename PointType> KOKKOS_INLINE_FUNCTION
key_type compute_key_for_point(const PointType& pos, const int depth) {
  LPM_KERNEL_ASSERT(depth>=0 && depth <= MAX_OCTREE_DEPTH);

  Real cx = 0; // coordinates of box centroid; root centroid = origin
  Real cy = 0; // coordinates of box centroid; root centroid = origin
  Real cz = 0; // coordinates of box centroid; root centroid = origin
  Real half_len = 1; // root box spans [-1,1] x [-1,1] x [-1,1]

  const auto nbits = 3*depth; // key length in bits
  key_type key = 0;
  for (auto i=1; i<=depth; ++i) {
    half_len *= 0.5;
    const auto b = nbits - 3*i; // position of level i's z-bit
    const bool rightz = pos(2) >= cz;
    const bool righty = pos(1) >= cy;
    const bool rightx = pos(0) >= cx;

    key += (rightz ? pow2<key_type>(b) : 0);
    cz  += (rightz ? half_len : -half_len);
    key += (righty ? pow2<key_type>(b+1) : 0);
    cy  += (righty ? half_len : -half_len);
    key += (rightx ? pow2<key_type>(b+2) : 0);
    cx  += (rightx ? half_len : -half_len);
  }
  return key;
}

/** @brief return the key of the parent of node k

  Mask all bits (turn off) at or below the kid's level.

  @param [in] kid_key key of child
  @param [in] level level of child
  @param [in] max_depth maximum depth of the octree
  @return key of parent
*/
KOKKOS_INLINE_FUNCTION
key_type parent_key(const key_type& kid_key, const int& level,
  const int& max_depth = MAX_OCTREE_DEPTH ) {

  LPM_KERNEL_ASSERT(max_depth > 0 && max_depth <= MAX_OCTREE_DEPTH);
  LPM_KERNEL_ASSERT(level > 0 && level <= MAX_OCTREE_DEPTH);

  const auto nbits = 3*max_depth; // key length in bits
  const auto pzb = nbits - 3*(level-1); // position of parent's z-bit
  key_type mask = 0;  // start by turning off all bits
  for (auto i=nbits; i>=pzb; --i) {
    mask += pow2<key_type>(i); // turn on bits at levels higher (lower depth) than level
  }
  // apply mask with bitwise and
  return key_type(kid_key & mask);
}

/** @brief given key k, return the local key (in [0,7]) of its node at level lev

  @param [in] k full key
  @param [in] lev level
  @param [in] max_depth maximum depth of the octree
  @return local key for level
*/
KOKKOS_INLINE_FUNCTION
key_type local_key(const key_type& k, const int& lev, const int& max_depth = MAX_OCTREE_DEPTH) {

  LPM_KERNEL_ASSERT(max_depth > 0 && max_depth <= MAX_OCTREE_DEPTH);
  LPM_KERNEL_ASSERT(lev > 0 && lev <= max_depth);

  const auto nbits = 3*max_depth; // key length in bits
  const auto pzb = nbits - 3*(lev-1); // postion of parent's z-bit
  key_type mask = 0; // start by turning off all bits
  for (auto i=pzb-3; i<pzb; ++i) {
    mask += pow2<key_type>(i); // turn on only the 3 bits for this level
  }
  key_type result = (k & mask); // apply mask
  return (result >> (pzb-3)); // shift results into [0,7]
}


/** @brief Construct the box associated with an octree node, given its key.

  @param [in] k key
  @param [in] lev level of node
  @param [in] max_depth maximum depth of octree
  @return box3d for node
*/
KOKKOS_INLINE_FUNCTION
Box3d box_from_key(const key_type& k, const int& lev, const int& max_depth=MAX_OCTREE_DEPTH) {
  LPM_KERNEL_ASSERT(max_depth > 0 && max_depth <= MAX_OCTREE_DEPTH);
  LPM_KERNEL_ASSERT(lev >= 0 && lev <= MAX_OCTREE_DEPTH);
  LPM_KERNEL_ASSERT(lev <= max_depth);
  Real cx = 0;
  Real cy = 0;
  Real cz = 0;
  Real half_len = 1;
  for (auto i=1; i<=lev; ++i) {
    half_len *= 0.5;
    const key_type lkey = local_key(k, i, max_depth);
    cz += ((lkey&1) > 0 ? half_len : -half_len);
    cy += ((lkey&2) > 0 ? half_len : -half_len);
    cx += ((lkey&4) > 0 ? half_len : -half_len);
  }
  return Box3d(cx-half_len, cx+half_len,
               cy-half_len, cy+half_len,
               cz-half_len, cz+half_len);
}

/** @brief Given the local child index (in [0,7]) and a parent key,
  return the key of the child node.

  @param [in] parent_key
  @param [in] kid_idx local index of child, relative to parent
  @param [in] kid_lev level of child
  @param [in] max_depth maximum depth of octree
  @return child node's key
*/
KOKKOS_INLINE_FUNCTION
key_type node_key(const key_type& parent_key, const key_type& kid_idx,
  const int& kid_lev, const int& max_depth) {
  LPM_KERNEL_ASSERT(max_depth > 0 && max_depth <= MAX_OCTREE_DEPTH);
  LPM_KERNEL_ASSERT(kid_lev > 0 && kid_lev <= MAX_OCTREE_DEPTH);
  const auto pzb = 3*max_depth - 3*(kid_lev-1);
  const key_type shifted_kid = (kid_idx << (pzb-3));
  return parent_key + shifted_kid;
}








} // namespace tree
} // namespace Lpm

#endif
