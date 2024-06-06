#ifndef LPM_MATH_UTIL_HPP
#define LPM_MATH_UTIL_HPP

#include "LpmConfig.h"
#include "lpm_constants.hpp"
#include "lpm_floating_point.hpp"
#include "lpm_tuple.hpp"
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
template <typename T>
KOKKOS_INLINE_FUNCTION const T& min(const T& a, const T& b) {
  return a < b ? a : b;
}

template <typename T>
KOKKOS_INLINE_FUNCTION const T& max(const T& a, const T& b) {
  return a > b ? a : b;
}

template <typename T>
KOKKOS_INLINE_FUNCTION const T* max_element(const T* const begin,
                                            const T* const end) {
  const T* me = begin;
  for (const T* it = begin + 1; it < end; ++it) {
    if (!(*it < *me)) me = it;
  }
  return me;
}

#else
using std::max;
using std::max_element;
using std::min;
#endif
using std::abs;

/// Inverse tangent with quadrant information, but with output range in [0,
/// 2*pi) instead of [-pi, pi)
KOKKOS_INLINE_FUNCTION
Real atan4(const Real y, const Real x) {
  Real result = 0.0;
  if (x == 0.0) {
    if (y > 0.0)
      result = 0.5 * constants::PI;
    else if (y < 0.0)
      result = 1.5 * constants::PI;
    else if (y == 0.0)
      result = 0.0;
  } else if (y == 0) {
    if (x > 0.0)
      result = 0.0;
    else if (x < 0.0)
      result = constants::PI;
  } else {
    Real theta = std::atan2(std::abs(y), std::abs(x));
    if (x > 0.0 && y > 0.0)
      result = theta;
    else if (x < 0.0 && y > 0.0)
      result = constants::PI - theta;
    else if (x < 0.0 && y < 0.0)
      result = constants::PI + theta;
    else if (x > 0.0 && y < 0.0)
      result = 2.0 * constants::PI - theta;
  }
  return result;
}

/// Determinant of a 2x2 matrix
template <typename T>
KOKKOS_INLINE_FUNCTION T two_by_two_determinant(const T a, const T b, const T c,
                                                const T d) {
  static_assert(std::is_arithmetic<T>::value,
                "two_by_two_determinant: arithmetic type required.");
  return a * d - b * c;
}

/// Quadratic formula
template <typename T>
KOKKOS_INLINE_FUNCTION void quadratic_roots(T& r1, T& r2, const T a, const T b,
                                            const T c) {
  static_assert(std::is_floating_point<T>::value,
                "quadratic formula: floating point type required.");
  const T two_a = 2 * a;
  T discrim = b * b - 4 * a * c;
  if (abs(discrim) <= constants::ZERO_TOL) {
    discrim = 0;
  } else if (discrim < -constants::ZERO_TOL) {
    // complex roots --- output garbage
    r1 = std::numeric_limits<T>::quiet_NaN();
    r2 = std::numeric_limits<T>::quiet_NaN();
  } else {
    const T sqrtdisc = sqrt(discrim);
    r1 = (-b + sqrtdisc) / two_a;
    r2 = (-b - sqrtdisc) / two_a;
  }
}

/// square a scalar
template <typename T = Real>
KOKKOS_INLINE_FUNCTION T square(const T& x) {
  static_assert(std::is_arithmetic<T>::value,
                "square: arithmetic type required.");
  return x * x;
}

/** safely divide by a real number

  @f$ \frac{1}{x} = \lim_{\epsilon \to 0} \frac{x}{x^2 + \epsilon^2} @f$

  @param x desired denominator
  @param eps regularization parameter
  @return @f$ \frac{x}{x^2 + \epsilon^2} @f$
*/
template <typename T = Real>
KOKKOS_INLINE_FUNCTION T safe_divide(const T& x, const T& eps = 1E-13) {
  static_assert(std::is_arithmetic<T>::value,
                "safe_divide: arithmetic type required.");
  return x / (square(x) + square(eps));
}

/// sgn function
template <typename T = Real>
KOKKOS_INLINE_FUNCTION T sign(const T& a) {
  static_assert(std::is_arithmetic<T>::value,
                "sign: arithmetic type required.");
  return (a > 0 ? 1 : (a < 0 ? -1 : 0));
}

/// Fortran's SIGN function
template <typename T = Real>
KOKKOS_INLINE_FUNCTION T sign(const T& a, const T& b) {
  static_assert(std::is_arithmetic<T>::value,
                "sign: arithmetic type required.");
  return a * sign(b);
}

/// cube a scalar
template <typename T = Real>
KOKKOS_INLINE_FUNCTION T cube(const T& x) {
  static_assert(std::is_arithmetic<T>::value,
                "cube: arithmetic type required.");
  return x * x * x;
}

/// rotation matrix to move an arbitrary point on the sphere to the north pole
template <typename Compressed3by3, typename PtType>
KOKKOS_INLINE_FUNCTION
void north_pole_rotation_matrix(Compressed3by3& rmat, const PtType& xyz) {
  const Real cosy = sqrt(square(xyz[1]) + square(xyz[2]));
  const Real siny = xyz[0];
  const bool on_x_axis = FloatingPoint<Real>::zero(cosy);
  const Real cosx = (on_x_axis ? 1 : xyz[2]/cosy);
  const Real sinx = (on_x_axis ? 0 : xyz[1]/cosy);
  rmat[0] =  cosy;
  rmat[1] = -sinx * siny;
  rmat[2] = -cosx * siny;
  rmat[3] =  0;
  rmat[4] =  cosx;
  rmat[5] = -sinx;
  rmat[6] =  siny;
  rmat[7] =  cosy * sinx;
  rmat[8] =  cosx * cosy;
}

template <typename Compressed3by3, typename PtType>
KOKKOS_INLINE_FUNCTION
void spherical_tangent_projection_matrix(Compressed3by3& mat, const PtType& x) {
  for (int i=0; i<3; ++i) {
    for (int j=0; j<3; ++j) {
      mat[3*i+j] = -x[i]*x[j] + (i==j ? 1 : 0);
    }
  }
}

template <typename PtType>
KOKKOS_INLINE_FUNCTION
Kokkos::Tuple<Real,9> spherical_tangent_projection_matrix(const PtType& x) {
  Kokkos::Tuple<Real,9> result;
  spherical_tangent_projection_matrix(result, x);
  return result;
}

template <typename PtType>
KOKKOS_INLINE_FUNCTION
Kokkos::Tuple<Real,9> north_pole_rotation_matrix(const PtType& xyz) {
  Kokkos::Tuple<Real,9> result;
  north_pole_rotation_matrix(result, xyz);
  return result;
}

template <typename PtType, typename Compressed3by3, typename ConstPtType>
KOKKOS_INLINE_FUNCTION
void apply_3by3(PtType& xyzp, const Compressed3by3& mat, const ConstPtType& xyz) {
  for (int i=0; i<3; ++i) {
    xyzp[i] = 0;
    for (int j=0; j<3; ++j) {
      xyzp[i] += mat[3*i + j]*xyz[j];
    }
  }
}

template <typename PtType, typename Compressed3by3, typename ConstPtType>
KOKKOS_INLINE_FUNCTION
void apply_3by3_transpose(PtType& xyzp, const Compressed3by3& mat, const ConstPtType& xyz) {
  for (int i=0; i<3; ++i) {
    xyzp[i] = 0;
    for (int j=0; j<3; ++j) {
      xyzp[i] += mat[3*j + i]*xyz[j];
    }
  }
}

template <typename MatType, typename ConstMatType1, typename ConstMatType2>
KOKKOS_INLINE_FUNCTION
void matmul_3by3(MatType& c, const ConstMatType1& a, const ConstMatType2& b) {
  for (int i=0; i<3; ++i) {
    for (int j=0; j<3; ++j) {
      const int c_idx = 3*i + j;
      c[c_idx] = 0;
      for (int k=0; k<3; ++k) {
          c[c_idx] += a[3*i+k] * b[3*k+j];
      }
    }
  }
}

template <typename Compressed3by3, typename PtType1, typename PtType2>
KOKKOS_INLINE_FUNCTION
void outer_product_r3(Compressed3by3& oprod, const PtType1& x, const PtType2& y) {
  for (int i=0; i<3; ++i) {
    for (int j=0; j<3; ++j) {
      oprod[3*i + j] = x[i] * y[j];
    }
  }
}

template <int ndim>
KOKKOS_INLINE_FUNCTION
void scal(Kokkos::Tuple<Real,ndim>& tuple, const Real a) {
  for (int i=0; i<ndim; ++i) {
    tuple[i] *= a;
  }
}

/** Return the Bessel function B_0(x) for any real number x.

 The polynomial approximation by
 series of Chebyshev polynomials is used for 0<X<8 and 0<8/X<1.

    REFERENCES:
     M.ABRAMOWITZ,I.A.STEGUN, HANDBOOK OF MATHEMATICAL FUNCTIONS, 1965.
     C.W.CLENSHAW, NATIONAL PHYSICAL LABORATORY MATHEMATICAL TABLES,
     VOL.5, 1962.
  @param [in] x
*/
template <typename T = Real>
KOKKOS_INLINE_FUNCTION T bessel_j0(const T& x) {
  static_assert(std::is_floating_point<T>::value, "floating point type required.");
  constexpr T p1 = 1.0;
  constexpr T p2 = -0.1098628627E-2;
  constexpr T p3 = 0.2734510407E-4;
  constexpr T p4 = -0.2073370639E-5;
  constexpr T p5 = 0.2093887211E-6;
  constexpr T q1 = -0.1562499995E-1;
  constexpr T q2 = 0.1430488765E-3;
  constexpr T q3 = -0.6911147651E-5;
  constexpr T q4 = 0.7621095161E-6;
  constexpr T q5 = -0.9349451520E-7;
  constexpr T r1 = 57568490574.0;
  constexpr T r2 = -13362590354.0;
  constexpr T r3 =  651619640.7;
  constexpr T r4 = -11214424.18;
  constexpr T r5 = 77392.33017;
  constexpr T r6 = -184.9052456;
  constexpr T s1 = 57568490411.0;
  constexpr T s2 = 1029532985.0;
  constexpr T s3 = 9494680.718;
  constexpr T s4 = 59272.64853;
  constexpr T s5 = 267.8532712;
  constexpr T s6 = 1.0;

  T result = 0;
  if (FloatingPoint<T>::zero(x)) {
    result = 1.0;
  }
  else {
    const T ax = abs(x);
    if (ax < 8) {
      const T y = x*x;
      const T fr = r1+y*(r2+y*(r3+y*(r4+y*(r5+y*r6))));
      const T fs = s1+y*(s2+y*(s3+y*(s4+y*(s5+y*s6))));
      result = fr/fs;
    }
    else {
      const T z = 8.0/ax;
      const T y = z*z;
      const T xx = ax - 0.785398164;
      const T fp = p1+y*(p2+y*(p3+y*(p4+y*p5)));
      const T fq = q1+y*(q2+y*(q3+y*(q4+y*q5)));
      result = sqrt(0.636619772/ax)*(fp*cos(xx)-z*fq*sin(xx));
    }
  }
  return result;
}

/** Return the Bessel function B_1(x) for any real number x.

 The polynomial approximation by
 series of Chebyshev polynomials is used for 0<X<8 and 0<8/X<1.

    REFERENCES:
     M.ABRAMOWITZ,I.A.STEGUN, HANDBOOK OF MATHEMATICAL FUNCTIONS, 1965.
     C.W.CLENSHAW, NATIONAL PHYSICAL LABORATORY MATHEMATICAL TABLES,
     VOL.5, 1962.
  @param [in] x
*/
template <typename T = Real>
KOKKOS_INLINE_FUNCTION T bessel_j1(const T& x) {
  static_assert(std::is_floating_point<T>::value, "floating point type required.");
  constexpr T p1 = 1.0;
  constexpr T p2 = 0.183105E-2;
  constexpr T p3 = -0.3516396496E-4;
  constexpr T p4 = 0.2457520174E-5;
  constexpr T p5 = -0.240337019E-6;
  constexpr T p6 = 0.636619772E0;
  constexpr T q1 = 0.04687499995E0;
  constexpr T q2 = -0.20026908730-3;
  constexpr T q3 = 0.8449199096E-5;
  constexpr T q4 = -0.88228987E-6;
  constexpr T q5 = 0.105787412E-6;
  constexpr T r1 = 72362614232.0;
  constexpr T r2 = -7895059235.0;
  constexpr T r3 = 242396853.1;
  constexpr T r4 = -2972611.439;
  constexpr T r5 = 15704.48260;
  constexpr T r6 = -30.16036606;
  constexpr T s1 = 144725228442.0;
  constexpr T s2 = 2300535178.0;
  constexpr T s3 = 18583304.74;
  constexpr T s4 = 99447.43394;
  constexpr T s5 = 376.9991397;
  constexpr T s6 = 1.0;

  T result = 0;
  const T abs_x = abs(x);
  if (abs_x < 8) {
    const T y = square(x);
    const T fr = r1+y*(r2+y*(r3+y*(r4+y*(r5+y*r6))));
    const T fs = s1+y*(s2+y*(s3+y*(s4+y*(s5+y*s6))));
    result = x*(fr/fs);
  }
  else {
    const T z = 8.0 / abs_x;
    const T y = square(z);
    const T xx = abs_x - 2.35619491;
    const T fp = p1+y*(p2+y*(p3+y*(p4+y*p5)));
    const T fq = q1+y*(q2+y*(q3+y*(q4+y*q5)));
    result = sqrt(p6/abs_x)*(cos(xx)*fp-z*sin(xx)*fq)*sign(s6,x);
  }
  return result;
}

}  // namespace Lpm

#endif
