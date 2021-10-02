#ifndef LPM_PYTHON3_UTILS_HPP
#define LPM_PYTHON3_UTILS_HPP

#include "LpmConfig.h"
#include <iostream>

namespace Lpm {

void write_numpy_import(std::ostream& os) {
  os << "import numpy as np\n\n";
}

template <typename ViewType>
void write_1d_numpy_array(std::ostream& os, const std::string& name, const ViewType& view) {
  auto v = Kokkos::create_mirror_view(view);
  const auto last_idx = v.extent(0)-1;
  Kokkos::deep_copy(v, view);
  os << name << " = np.array([";
  for (Index i=0; i<last_idx; ++i) {
    os << v(i) << ",";
  }
  os << v(last_idx) << "])\n";
}

template <typename ViewType>
void write_2d_numpy_array(std::ostream& os, const std::string& name, const ViewType& view) {
  auto v = Kokkos::create_mirror_view(view);
  Kokkos::deep_copy(v, view);
  const auto last_row_idx = v.extent(0)-1;
  const auto last_col_idx = v.extent(1)-1;
  os << name << "_entries = np.array([";
  for (Index i=0; i<last_row_idx; ++i) {
    for (Index j=0; j<=last_col_idx; ++j) {
      os << v(i,j) << ",";
    }
  }
  for (Index j=0; j<last_col_idx; ++j) {
    os << v(last_row_idx,j) << ",";
  }
  os << v(last_row_idx, last_col_idx) << "])\n";
  os << name << " = np.reshape(" << name << "_entries, (" << v.extent(0) << ","
     << v.extent(1) << "), order='C')\n";
}

class PythonModule {
  int tab_level;
  bool is_init;

  std::ostream& os;

  void init();

  public:
    PythonModule(std::ostream& s = std::cout) :
      tab_level(0),
      is_init(false),
      os(s) {init();}

   void dataset_init(const std::string& dataset_name);

};



} // namespace Lpm

#endif

