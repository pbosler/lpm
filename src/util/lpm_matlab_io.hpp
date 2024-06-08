#ifndef LPM_MATLAB_IO_HPP
#define LPM_MATLAB_IO_HPP

#include <iostream>

#include "LpmConfig.h"

namespace Lpm {

template <typename HVT>
inline void write_vector_matlab(std::ostream& os, const std::string& name,
                         const HVT& v) {
  static_assert(Kokkos::SpaceAccessibility<typename HVT::execution_space,
                                           Kokkos::HostSpace>::accessible,
                "HostSpace required for i/o.");
  const auto last_idx = v.extent(0) - 1;
  os << name << " = [";
  for (Index i = 0; i < last_idx; ++i) {
    os << v(i) << ",";
  }
  os << v(last_idx) << "];\n";
}

template <>
inline void write_vector_matlab<std::vector<Real>>(std::ostream& os,
  const std::string& name, const std::vector<Real>& v) {
  const auto last_idx = v.size()-1;
  os << name << " = [";
  for (Index i=0; i<last_idx; ++i) {
    os << v[i] << ",";
  }
  os << v[last_idx] << "];\n";
  }

template <typename HVT>
inline void write_array_matlab(std::ostream& os, const std::string name, const HVT a) {
  static_assert(Kokkos::SpaceAccessibility<typename HVT::execution_space,
                                           Kokkos::HostSpace>::accessible,
                "HostSpace required for i/o.");
  const auto nrow = a.extent(0);
  const auto last_row = nrow - 1;
  const auto ncol = a.extent(1);
  const auto last_col = ncol - 1;
  os << name << " = [";
  for (Index i = 0; i < nrow; ++i) {
    for (Index j = 0; j < ncol; ++j) {
      os << a(i, j)
         << (i < last_row ? (j < last_col ? "," : ";")
                          : (j < last_col ? "," : "];\n"));
    }
  }
}

}  // namespace Lpm

#endif
