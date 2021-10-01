#ifndef LPM_QUADTREE_LOOKUP_TABLES_HPP
#define LPM_QUADTREE_LOOKUP_TABLES_HPP

#include "LpmConfig.h"
#include "lpm_assert.hpp"
#include "tree/lpm_quadtree_functions.hpp"
#include <array>

namespace Lpm {
namespace quadtree {

std::array<Int,36> parent_lookup_table_entries();

std::array<Int,36> child_lookup_table_entries();

struct ParentLUT {
  const Int nrows = 4;
  const Int ncols = 9;
  const Int entries[36] = {0, 1, 1, 3, 4, 4, 3, 4, 4,
                           1, 1, 2, 4, 4, 5, 4, 4, 5,
                           3, 4, 4, 3, 4, 4, 6, 7, 7,
                           4, 4, 5, 4, 4, 5, 7, 7, 8};
};

struct ChildLUT {
  const Int nrows = 4;
  const Int ncols = 9;
  const Int entries[36] = {3, 2, 3, 1, 0, 1, 3, 2, 3,
                           2, 3, 2, 0, 1, 0, 2, 3, 2,
                           1, 0, 1, 3, 2, 3, 1, 0, 1,
                           0, 1, 0, 2, 3, 2, 0, 1, 0};
};

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

} // namespace quadtree
} // namespace Lpm

#endif
