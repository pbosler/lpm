#ifndef LPM_TREE_COMMON_HPP
#define LPM_TREE_COMMON_HPP

#include "LpmConfig.h"
#include "lpm_assert.hpp"

namespace Lpm {

template <typename TableViewType> KOKKOS_INLINE_FUNCTION
Int table_val(const Int& i, const Int& j, const TableViewType tableview) {
    LPM_KERNEL_ASSERT(i >=0 && i < tableview().nrows);
    LPM_KERNEL_ASSERT(j >=0 && j < tableview().ncols);
    return tableview().entries[tableview().ncols*i+j];
}

template <typename TableType> KOKKOS_INLINE_FUNCTION
Int table_val(const Int& i, const Int& j, const Kokkos::View<TableType>& tableview) {
  return table_val<Kokkos::View<TableType>>(i, j, tableview);
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



} // namespace Lpm

#endif
