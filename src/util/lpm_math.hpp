#ifndef LPM_MATH_UTIL_HPP
#define LPM_MATH_UTIL_HPP

#include "LpmConfig.h"
#include "lpm_constants.hpp"
#ifdef LPM_USE_BOOST
#include "boost/math/special_functions/bessel.hpp"
#include "boost/math/special_functions/legendre.hpp"
#endif
#include <algorithm>
#include <cmath>
#include <limits>
#include "Kokkos_Core.hpp"

namespace Lpm {

#ifdef KOKKOS_ENABLE_CUDA
/// GPU-friendly replacements for stdlib functions
template <typename T> KOKKOS_INLINE_FUNCTION
const T& min (const T& a, const T& b) {return a < b ? a : b;}

template <typename T> KOKKOS_INLINE_FUNCTION
const T& max (const T& a, const T& b) {return a > b ? a : b;}

template <typename T> KOKKOS_INLINE_FUNCTION
const T* max_element(const T* const begin, const T* const end) {
    const T* me = begin;
    for (const T* it=begin +1; it < end; ++it) {
        if (!(*it < *me)) me = it;
    }
    return me;
}

#else
using std::min;
using std::max;
using std::max_element;
#endif
using std::abs;

#ifdef LPM_USE_BOOST
KOKKOS_INLINE_FUNCTION
Real cyl_bessel_j(const int j, const Real x) {
  return boost::math::cyl_bessel_j<int, Real>(j, x);
}

KOKKOS_INLINE_FUNCTION
Real legendre_p(const int l, const Real z) {
  return boost::math::legendre_p<Real>(l, z);
}

KOKKOS_INLINE_FUNCTION
Real legendre_p(const int l, const int m, const Real z) {
  return boost::math::legendre_p<Real>(l,m,z);
}
#endif


/// Inverse tangent with quadrant information, but with output range in [0, 2*pi) instead of (-pi, pi]
KOKKOS_INLINE_FUNCTION
Real atan4(const Real y, const Real x){
  Real result = 0.0;
if ( x == 0.0 )
{
  if ( y > 0.0 )
    result = 0.5 * constants::PI;
  else if ( y < 0.0 )
    result = 1.5 * constants::PI;
  else if ( y == 0.0 )
    result = 0.0;
}
else if ( y == 0 )
{
  if ( x > 0.0 )
    result = 0.0;
  else if ( x < 0.0 )
    result = constants::PI;
}
else
{
  Real theta = std::atan2( std::abs(y), std::abs(x) );
  if ( x > 0.0 && y > 0.0 )
    result = theta;
  else if ( x < 0.0 && y > 0.0 )
    result = constants::PI - theta;
  else if ( x < 0.0 && y < 0.0 )
    result = constants::PI + theta;
  else if ( x > 0.0 && y < 0.0 )
    result = 2.0 * constants::PI - theta;
}
return result;
}

/// Determinant of a 2x2 matrix
template <typename T> KOKKOS_INLINE_FUNCTION
T two_by_two_determinant(const T a, const T b, const T c, const T d) {
  static_assert(std::is_arithmetic<T>::value, "two_by_two_determinant: arithmetic type required.");
  return a*d - b*c;
}

/// Quadratic formula
template <typename T> KOKKOS_INLINE_FUNCTION
void quadratic_roots(T& r1, T& r2, const T a, const T b, const T c) {
  static_assert(std::is_floating_point<T>::value, "quadratic formula: floating point type required.");
  const T two_a = 2*a;
  T discrim = b*b - 4*a*c;
  if (abs(discrim) <= constants::ZERO_TOL) {
    discrim = 0;
  }
  else if (discrim < -constants::ZERO_TOL) {
    // complex roots --- output garbage
    r1 = std::numeric_limits<T>::quiet_NaN();
    r2 = std::numeric_limits<T>::quiet_NaN();
  }
  else {
    const T sqrtdisc = sqrt(discrim);
    r1 = (-b + sqrtdisc) / two_a;
    r2 = (-b - sqrtdisc) / two_a;
  }
}

/** safely divide by a real number

  @f$ \frac{1}{x} = \lim_{\epsilon \to 0} \frac{x}{x^2 + \epsilon^2} @f$

  @param x desired denominator
  @param eps regularization parameter
  @return @f$ \frac{x}{x^2 + \epsilon^2} @f$
*/
template <typename T=Real> KOKKOS_INLINE_FUNCTION
T safe_divide(const T& x, const T& eps=1E-13) {return x/(square(x) + square(eps));}

/// square a scalar
template <typename T=Real> KOKKOS_INLINE_FUNCTION
T square(const T& x) {
  static_assert(std::is_arithmetic<T>::value, "square: arithmetic type required.");
  return x*x;
}
/// sgn function
template <typename T=Real> KOKKOS_INLINE_FUNCTION
T sign(const T& a) {
  static_assert(std::is_arithmetic<T>::value, "sign: arithmetic type required.");
  return (a>0 ? 1 : (a < 0 ? -1 : 0));
}
/// cube a scalar
template <typename T=Real> KOKKOS_INLINE_FUNCTION
T cube(const T& x) {
  static_assert(std::is_arithmetic<T>::value, "cube: arithmetic type required.");
  return x*x*x;
}

} // namespace Lpm

#endif
