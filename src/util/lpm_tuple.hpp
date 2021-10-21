#ifndef LPM_TUPLE_HPP
#define LPM_TUPLE_HPP

#include "LpmConfig.h"
#include "Kokkos_Core.hpp"
#include "Kokkos_Array.hpp"
#include "spdlog/fmt/ostr.h"
#include "util/lpm_math.hpp"
#include "util/lpm_floating_point.hpp"
#include <limits>
#include <cfloat>
/**
Kokkos-array subclass for reductions
*/
namespace Kokkos {

/** \brief Tuple type for || reduction.

  T is a plain old data type
  ndim is the number of T's in the tuple.

  Basic functions handled by superclass, Kokkos::Array
  This subclass adds the required operators for sum and product reductions.
*/
template <typename T, int ndim> struct Tuple : public Array<T,ndim> {
  KOKKOS_FORCEINLINE_FUNCTION
  Tuple() : Array<T,ndim>() {
    for (int i=0; i<ndim; ++i)
      this->m_internal_implementation_private_member_data[i] = 0;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  Tuple(const Tuple<T,ndim>& val) : Array<T,ndim>() {
    for (int i=0; i<ndim; ++i)
      this->m_internal_implementation_private_member_data[i] = val[i];
  }

  KOKKOS_FORCEINLINE_FUNCTION
  Tuple(const T& v0, const T& v1) : Array<T,ndim>() {
    this->m_internal_implementation_private_member_data[0] = v0;
    this->m_internal_implementation_private_member_data[1] = v1;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  Tuple(const T& v0, const T& v1, const T& v2) : Array<T,ndim>() {
    this->m_internal_implementation_private_member_data[0] = v0;
    this->m_internal_implementation_private_member_data[1] = v1;
    this->m_internal_implementation_private_member_data[2] = v2;
  }

  KOKKOS_INLINE_FUNCTION
  Tuple(const volatile Tuple<T, ndim>& src) : Array<T, ndim>() {
    const auto n = std::min(this->size(), src.size());
    for (auto i=0; i<n; ++i) this->m_internal_implementation_private_member_data[i] = src[i];
  }

  KOKKOS_FORCEINLINE_FUNCTION
  Tuple(const T* ptr) : Array<T,ndim>() {
    for (int i=0; i<ndim; ++i) {
      this->m_internal_implementation_private_member_data[i] = ptr[i];
    }
  }

  KOKKOS_FORCEINLINE_FUNCTION
  Tuple(const View<T*>& v) : Array<T,ndim>() {
    for (int i=0; i<ndim; ++i) {
      this->m_internal_implementation_private_member_data[i] = v[i];
    }
  }

  KOKKOS_INLINE_FUNCTION
  volatile T& operator[] (const int& i) volatile {
    return this->m_internal_implementation_private_member_data[i];}

  KOKKOS_INLINE_FUNCTION
  volatile typename std::add_const<T>::type & operator[] (const int& i) const volatile {
    return this->m_internal_implementation_private_member_data[i];
  }

  KOKKOS_FORCEINLINE_FUNCTION
  Tuple operator += (const Tuple<T,ndim>& o) {
    for (int i=0; i<ndim; ++i)
      this->m_internal_implementation_private_member_data[i] += o[i];
    return *this;
  }
  KOKKOS_INLINE_FUNCTION
  Tuple operator += (const volatile Tuple<T,ndim>& o) volatile {
    for (int i=0; i<ndim; ++i)
      this->m_internal_implementation_private_member_data[i] += o[i];
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  Tuple& operator=(const Tuple<T,ndim>& rhs) {
    const auto n = std::min(this->size(), rhs.size());
    for (auto i=0; i<n; ++i) this->m_internal_implementation_private_member_data[i] = rhs[i];
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  Tuple operator=(const volatile Tuple<T,ndim>& rhs) {
    const auto n = std::min(this->size(), rhs.size());
    for (auto i=0; i<n; ++i) this->m_internal_implementation_private_member_data[i] = rhs[i];
    return *this;
  }



  KOKKOS_FORCEINLINE_FUNCTION
  Tuple operator *= (const Tuple<T,ndim>& o) {
    for (int i=0; i<ndim; ++i)
      this->m_internal_implementation_private_member_data[i] *= o[i];
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  Tuple operator *= (const volatile Tuple<T,ndim>& o) volatile {
    for (int i=0; i<ndim; ++i)
      this->m_internal_implementation_private_member_data[i] *= o[i];
    return *this;
  }

  template <typename OStream>
  friend OStream& operator << (OStream& os, const Tuple<T,ndim>& tup) {
    os << "[ ";
    for (int i=0; i<ndim; ++i) {
      os << tup[i] << " ";
    }
    os << "]";
    return os;
  }
};

template <int ndim>
struct reduction_identity<Tuple<Lpm::Real,ndim>> {
  KOKKOS_FORCEINLINE_FUNCTION static Tuple<Lpm::Real,ndim> sum() {return Tuple<Lpm::Real,ndim>();}
  KOKKOS_FORCEINLINE_FUNCTION static Tuple<Lpm::Real,ndim> prod() {return Tuple<Lpm::Real,ndim>(1);}
  KOKKOS_FORCEINLINE_FUNCTION static Tuple<Lpm::Real,ndim> max() {return Tuple<Lpm::Real,ndim>(-DBL_MAX);}
  KOKKOS_FORCEINLINE_FUNCTION static Tuple<Lpm::Real,ndim> min() {return Tuple<Lpm::Real,ndim>(DBL_MIN);}
};

template <typename T, int ndim> KOKKOS_INLINE_FUNCTION
typename std::enable_if<std::is_floating_point<T>::value, bool>::type
operator == (const Tuple<T,ndim>& lhs, const Tuple<T,ndim>& rhs) {
  bool result = true;
  for (int i=0; i<ndim; ++i) {
    if (!Lpm::FloatingPoint<T>::equiv(lhs[i], rhs[i])) result = false;
  }
  return result;
}

template <typename T, int ndim> KOKKOS_INLINE_FUNCTION
typename std::enable_if<std::is_integral<T>::value, bool>::type
operator == (const Tuple<T,ndim>& lhs, const Tuple<T,ndim>& rhs) {
  bool result = true;
  for (int i=0; i<ndim; ++i) {
    if (lhs[i] != rhs[i]) result = false;
  }
  return result;
}

template <typename T, int ndim> KOKKOS_INLINE_FUNCTION
bool operator != (const Tuple<T, ndim>& lhs, const Tuple<T,ndim>& rhs) {
  return !(lhs == rhs);
}

}// namespace Kokkos

namespace Lpm {

}
#endif
