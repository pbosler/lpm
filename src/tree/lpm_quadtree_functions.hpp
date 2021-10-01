#ifndef LPM_QUADTREE_FUNCTIONS_HPP
#define LPM_QUADTREE_FUNCTIONS_HPP

#include "LpmConfig.h"
#include "lpm_kokkos_defs.hpp"
#include "tree/lpm_box2d.hpp"

namespace Lpm {
namespace quadtree {

#define MAX_QUADTREE_DEPTH 16
#define WORD_MASK 0xFFFFFFFF
#define NULL_IDX -1

typedef uint32_t key_type;
typedef uint_fast32_t id_type;
typedef uint_fast64_t code_type;

template <typename PointType, typename Itype=short> KOKKOS_INLINE_FUNCTION
key_type compute_key_for_point(const PointType& pos, const Itype depth,
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

template <typename IType=short> KOKKOS_INLINE_FUNCTION
key_type parent_key(const key_type& kid_key, const IType lev, const IType max_depth) {
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

template <typename IType=short> KOKKOS_INLINE_FUNCTION
key_type local_key(const key_type& key, const IType lev, const IType max_depth) {
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

template <typename IType=short> KOKKOS_INLINE_FUNCTION
Box2d box_from_key(const key_type& k, const IType lev, const IType max_depth) {
  LPM_KERNEL_ASSERT(max_depth > 0 && max_depth <= MAX_QUADTREE_DEPTH);
  LPM_KERNEL_ASSERT( lev >= 0 && lev <= max_depth);
  Real cx = 0;
  Real cy = 0;
  Real half_len = 1;
  for (auto i=1; i<=lev; ++i) {
    half_len *= 0.5;
    const key_type lkey = local_key(k, i, max_depth);
    cy += ( (lkey&1) > 0 ? half_len : -half_len);
    cx += ( (lkey&2) > 0 ? half_len : -half_len);
  }
  return Box2d(cx - half_len, cx + half_len,
               cy - half_len, cy + half_len);
}

/** @brief given a point's global index and its node key,
  compute its packed sorting code so that the
  node key is in the first 32 bits and the point id is in the second 32 bits
  of the code.

  @param [in] key key of leaf node containing point
  @param [in] id global id of point
  @return sorting code
*/
KOKKOS_INLINE_FUNCTION
code_type encode(const key_type key, const id_type id) {
  code_type result(key);
  return ((result<<32) + id);
}

/** @brief give a packed 64-bit sort code recover the encoded point id
  from the lowest 32 bits.
*/
KOKKOS_INLINE_FUNCTION
id_type decode_id(const code_type& code) {
  return id_type(code);
}

/** @brief give a packed 64-bit sort code recover the encoded node key
  from the highest 32 bits.
*/
KOKKOS_INLINE_FUNCTION
key_type decode_key(const code_type& code) {
  return key_type(code>>32);
}

template <typename T, typename T2>
struct Converter {
  KOKKOS_INLINE_FUNCTION
  Converter() {}

  KOKKOS_INLINE_FUNCTION
  T operator() (const T2 src) const {
    return T(src);
  }
};

template <>
struct Converter<key_type, code_type> {
  KOKKOS_INLINE_FUNCTION
  Converter() {}

  KOKKOS_INLINE_FUNCTION
  key_type operator() (const code_type c) const {
    return decode_key(c);
  }
};

/** @brief search a sorted view for a key

  @param [in] key
  @param [in] sorted_view
  @return index of key in sorted_view
*/
template <typename T=key_type, typename ViewType> KOKKOS_INLINE_FUNCTION
Index binary_search_first(const T& target, const ViewType& sorted_view) {
  Index low = 0;
  Index high = sorted_view.extent(0)-1;
  Index result = NULL_IDX;
  Converter<T, typename ViewType::value_type> convert;
  while ( low <= high) {
    Index mid = (low + high)/2;
    const auto midval = convert(sorted_view(mid));
    if (target == midval) {
      result = mid;
      high = mid-1;
    }
    else if (target < midval) {
      high = mid-1;
    }
    else {
      low = mid+1;
    }
  }
  return result;
}

template <typename T=key_type, typename ViewType> KOKKOS_INLINE_FUNCTION
Index binary_search_last(const T& target, const ViewType& sorted_view) {
  Index low = 0;
  Index high = sorted_view.extent(0)-1;
  Index result = NULL_IDX;
  Converter<T, typename ViewType::value_type> convert;
  while (low <= high) {
    Index mid = (low + high)/2;
    const auto midval = convert(sorted_view(mid));
    if (target == midval) {
      result = mid;
      low = mid+1;
    }
    else if (target < midval) {
      high = mid-1;
    }
    else {
      low = mid+1;
    }
  }
  return result;
}


} // namespace quadtree
} // namespace Lpm

#endif
