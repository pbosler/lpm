#ifndef LPM_FIELD_IMPL_HPP
#define LPM_FIELD_IMPL_HPP

#include "lpm_field.hpp"
#include "util/lpm_string_util.hpp"
#include <sstream>

namespace Lpm {

template <FieldLocation FL>
std::pair<Real, Real> ScalarField<FL>::range(const Index n) const {
 typename Kokkos::MinMax<Real>::value_type min_max;
 auto vals = view;
 Kokkos::parallel_reduce("scalar field range", n,
    KOKKOS_LAMBDA (const Index i, typename Kokkos::MinMax<Real>::value_type& mm) {
    if (vals(i) < mm.min_val) mm.min_val = vals(i);
    if (vals(i) > mm.max_val) mm.max_val = vals(i);
    }, Kokkos::MinMax<Real>(min_max));
  return std::make_pair(min_max.min_val, min_max.max_val);
}

template <FieldLocation FL>
std::string ScalarField<FL>::info_string(const int tab_level) const {
  std::ostringstream ss;
  auto tabstr = indent_string(tab_level);
  ss << tabstr << "ScalarField info:\n";
  tabstr += "\t";
  for (auto& md : metadata) {
    ss << tabstr << md.first << ": " << md.second << "\n";
  }
  const auto r = range(view.extent(0));
  ss << tabstr << "range: [" << r.first << ", " << r.second << "]\n";
  return ss.str();
};

template <typename Geo, FieldLocation FL>
std::string VectorField<Geo, FL>::info_string(const int tab_level) const {
  std::ostringstream ss;
  auto tabstr = indent_string(tab_level);
  ss << tabstr << "VectorField info:\n";
  tabstr += "\t";
  for (auto& md : metadata) {
    ss << tabstr << md.first << ": " << md.second << "\n";
  }
  return ss.str();
};

} // namespace Lpm

#endif
