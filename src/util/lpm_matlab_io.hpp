#ifndef LPM_MATLAB_IO_HPP
#define LPM_MATLAB_IO_HPP

#include "LpmConfig.h"
#include <iostream>
#include <string>

namespace Lpm {

template <typename ViewType>
void write_vector_matlab(std::ostream& os, const std::string name, const ViewType& v) {
  os << name << " = [";
  for (Index i=0; i<v.extent(0)-1; ++i)
      os << v(i) << ",";
  os << v(v.extent(0)-1) << "];\n";
}

template <typename ViewType>
void write_array_matlab(std::ostream& os, const std::string name, const ViewType& a) {
  os << name << " = [";
  const Index nrow = a.extent(0);
  const Int ncol = a.extent(1);
  for (Index i=0; i<nrow; ++i) {
      for (Int j=0; j<ncol; ++j) {
          os << a(i,j) << (i<nrow-1 ? (j<ncol-1 ? "," : ";") : (j<ncol-1 ? "," : "];\n"));
      }
  }
}

template <typename BoxViewType>
void write_box_verts_matlab(std::ostream& os, const BoxViewType& boxes) {
  static_assert(std::is_same<typename BoxViewType::value_type, Box2d>::value,
    "only Box2d supported.");

  auto h_boxes = Kokkos::create_mirror_view(boxes);
  Kokkos::deep_copy(h_boxes, boxes);

  os << "box_vert_crds = [";
  for (auto i=0; i<boxes.extent(0); ++i) {
    for (auto v=0; v<4; ++v) {
      const auto xy = h_boxes(i).vertex_crds(v);
      os << xy[0] << "," << xy[1] << (i<boxes.extent(0)-1 or v != 3 ? ";" : "];\n");
    }
  }
  os << "box_verts = [";
  for (auto i=0; i<boxes.extent(0); ++i) {
    for (auto v=0; v<4; ++v) {
      os << 4*i + v + 1 << (i==boxes.extent(0)-1 ? (v==3 ? "];\n" : ",") : (v==3 ? ";" : ","));
    }
  }
}

}
#endif
