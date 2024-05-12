#ifndef LPM_FIELD_IMPL_HPP
#define LPM_FIELD_IMPL_HPP

#include "lpm_field.hpp"
#include "util/lpm_string_util.hpp"
#include <cmath>
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

template <typename Geo, FieldLocation FL>
std::pair<Real, Real> VectorField<Geo, FL>::range(const Index n) const {
  typename Kokkos::MinMax<Real>::value_type min_max;
  auto vals = view;
  Kokkos::parallel_reduce("vector field range", n,
    KOKKOS_LAMBDA (const Index i, typename Kokkos::MinMax<Real>::value_type& mm) {
      const auto veci = Kokkos::subview(vals, i, Kokkos::ALL);
      const Real magi = Geo::mag(veci);
      if (magi < mm.min_val) mm.min_val = magi;
      if (magi > mm.max_val) mm.max_val = magi;
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

template <FieldLocation FL>
Index ScalarField<FL>::nan_count(const Index n) const {
  auto vals = view;
  Index nan_count = 0;
  Kokkos::parallel_reduce("scalar field has_nan", n,
    KOKKOS_LAMBDA (const Index i, Index& ct) {
      ct += (std::isfinite(vals(i)) ? 0 : 1);
    }, nan_count);
  return nan_count;
}

template <FieldLocation FL>
bool ScalarField<FL>::has_nan(const Index n) const {
  return (nan_count(n) > 0);
}

template <typename Geo, FieldLocation FL>
Index VectorField<Geo, FL>::nan_count(const Index n) const {
  auto vals = view;
  Index nan_count = 0;
  auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {n, Geo::ndim});
  Kokkos::parallel_reduce(policy,
    KOKKOS_LAMBDA (const Index i, const Int j, Index& ct) {
      ct += (std::isfinite(vals(i,j)) ? 0 : 1);
    }, nan_count);
  return nan_count;
}

template <typename Geo, FieldLocation FL>
bool VectorField<Geo,FL>::has_nan(const Index n) const {
  return (nan_count(n) > 0);
}

template <typename Geo, FieldLocation FL>
std::string VectorField<Geo, FL>::info_string(const int tab_level) const {
  std::ostringstream ss;
  auto tabstr = indent_string(tab_level);
  ss << tabstr << "VectorField info:\n";
  tabstr += "\t";
  for (auto& md : metadata) {
    ss << tabstr << md.first << ": " << md.second << "\n";
  }
  const auto r = range(view.extent(0));
  ss << tabstr << "range: [" << r.first << ", " << r.second << "]\n";
  return ss.str();
};

} // namespace Lpm

#endif

