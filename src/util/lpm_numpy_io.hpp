#ifndef LPM_NUMPY_IO_HPP
#define LPM_NUMPY_IO_HPP

#include <iostream>

#include "LpmConfig.h"

namespace Lpm {

void numpy_import(std::ostream& os) { os << "import numpy as np\n"; }

template <typename HVT>
void write_vector_numpy(std::ostream& os, const std::string name, const HVT v) {
  static_assert(Kokkos::SpaceAccessibility<typename HVT::execution_space,
                                           Kokkos::HostSpace>::accessible,
                "HostSpace required for i/o.");
  const auto last_idx = v.extent(0) - 1;
  os << name << " = np.array([";
  for (Index i = 0; i < last_idx; ++i) {
    os << v(i) << ",";
  }
  os << v(last_idx) << "])\n";
}

template <typename HVT>
void write_array_numpy(std::ostream& os, const std::string name, const HVT a) {
  static_assert(Kokkos::SpaceAccessibility<typename HVT::execution_space,
                                           Kokkos::HostSpace>::accessible,
                "HostSpace required for i/o.");
  const auto nrow = a.extent(0);
  const auto last_row = nrow - 1;
  const auto ncol = a.extent(1);
  const auto last_col = ncol - 1;
  os << name << " = np.array([";
  for (Index i = 0; i < nrow; ++i) {
    for (Index j = 0; j < ncol; ++j) {
      os << a(i, j)
         << (i < last_row ? (j < last_col ? "," : ";")
                          : (j < last_col ? "," : "])\n"));
    }
  }
}

}  // namespace Lpm
#endif
