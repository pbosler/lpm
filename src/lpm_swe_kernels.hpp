#ifndef LPM_SWE_KERNELS_HPP
#define LPM_SWE_KERNELS_HPP

#include "LpmConfig.h"
#include "lpm_geometry.hpp"
#include "util/lpm_math.hpp"
#include "lpm_assert.hpp"

namespace Lpm {

namespace impl {

template <typename UType, typename XType, typename YType>
KOKKOS_INLINE_FUNCTION void
kzeta_sphere(UType &u, const XType &x, const YType &y, const Real vort_y,
             const Real area_y, const Real eps = 0) {
  const Real denom =
      4 * constants::PI * (1 - SphereGeometry::dot(x, y) + square(eps));
  const Real strength = -vort_y * area_y / denom;
  Real uloc[3];
  SphereGeometry::cross(uloc, x, y);
  for (Short j = 0; j < 3; ++j) {
    u[j] = uloc[j] * strength;
  }
}

template <typename UType, typename XType, typename YType>
KOKKOS_INLINE_FUNCTION void
kzeta_plane(UType& u, const XType& x, const YType& y, const Real vort_y, const Real area_y, const Real eps = 0) {
  Real xmy[2];
  xmy[0] = x[0] - y[0];
  xmy[1] = x[1] - y[1];
  const Real denom = 2*constants::PI * (PlaneGeometry::norm2(xmy) + square(eps));
  const Real strength = vort_y * area_y / denom;
  u[0] = -xmy[1] * strength;
  u[1] =  xmy[0] * strength;
}

template <typename UType, typename XType, typename YType>
KOKKOS_INLINE_FUNCTION void
ksigma_plane(UType& u, const XType& x, const YType& y, const Real div_y, const Real area_y, const Real eps = 0) {
  Real xmy[2];
  xmy[0] = x[0] - y[0];
  xmy[1] = x[1] - y[1];
  const Real denom = 2*constants::PI * (PlaneGeometry::norm2(xmy) + square(eps));
  const Real strength = div_y * area_y / denom;
  u[0] = xmy[0] * strength;
  u[1] = xmy[1] * strength;
}

template <typename UType, typename XType, typename YType>
KOKKOS_INLINE_FUNCTION void
ksigma_sphere(UType &u, const XType &x, const YType &y, const Real div_y,
              const Real area_y, const Real eps = 0) {
  const Real denom =
      4 * constants::PI * (1 - SphereGeometry::dot(x, y) + square(eps));
  const Real strength = div_y * area_y / denom;
  Real uloc[3];
  Real pmat[3];
  for (Short j = 0; j < 3; ++j) {
    uloc[j] = 0;
    SphereGeometry::proj_row(pmat, x, j);
    for (Short k = 0; k < 3; ++k) {
      uloc[j] += pmat[k] * y[k];
    }
    u[j] = uloc[j] * strength;
  }
}

template <typename Matrix2by2, typename XType, typename YType>
KOKKOS_INLINE_FUNCTION
void grad_kzeta_plane(Matrix2by2& gkz, const XType& x, const YType& y, const Real eps = 0) {
  Real xmy[2];
  xmy[0] = x[0] - y[0];
  xmy[1] = x[1] - y[1];
  const Real epssq = square(eps);
  const Real denom = 1.0/(2*constants::PI * square((PlaneGeometry::norm2(xmy) + epssq)));
  gkz[0] =  2*xmy[0] * xmy[1]; // matrix 1,1
  gkz[1] =  square(xmy[0]) - square(xmy[1]) + epssq; // matrix 1, 2
  gkz[2] = -square(xmy[0]) + square(xmy[1]) + epssq; // matrix 2, 1
  gkz[3] = -2*xmy[0] * xmy[1]; // matrix 2,2
  for (short i=0; i<4; ++i) {
    gkz[i] *= denom;
  }
}

template <typename Matrix2by2, typename XType, typename YType>
KOKKOS_INLINE_FUNCTION
void grad_ksigma_plane(Matrix2by2& gks, const XType& x, const YType& y, const Real eps=0) {
  Real xmy[2];
  xmy[0] = x[0] - y[0];
  xmy[1] = x[1] - y[1];
  const Real epssq = square(eps);
  const Real denom = 1.0/(2*constants::PI * square((PlaneGeometry::norm2(xmy) + epssq)));
  gks[0] = -square(xmy[0]) + square(xmy[1]) + epssq;
  gks[1] = -2*xmy[0]*xmy[1];
  gks[2] = -2*xmy[0]*xmy[1];
  gks[3] =  square(xmy[0]) - square(xmy[1]) + epssq;
  for (short i=0; i<4; ++i) {
    gks[i] *= denom;
  }
}

template <typename Matrix3by3, typename XType, typename YType>
KOKKOS_INLINE_FUNCTION
void grad_kzeta_sphere(Matrix3by3 &gkz, const XType &x,
                                       const YType &y, const Real eps = 0) {
  const Real epssq = square(eps);
  const Real denom =
      1.0 / (4 * constants::PI * square(1 - SphereGeometry::dot(x, y) + epssq));

  gkz[0] = ((1 + epssq) * x[0] - y[0]) *
           (x[1] * y[2] - x[2] * y[1]); // grad kzeta matrix 1,1
  gkz[1] = (1 + epssq) * (-(1 - square(x[1])) * y[2] - x[1] * x[2] * y[1]) +
           x[0] * y[0] * y[2] +
           x[2] * (square(y[1]) + square(y[2])); // grad kzeta matrix 1,2
  gkz[2] = (1 + epssq) * ((1 - square(x[2])) * y[1] + x[1] * x[2] * y[2]) -
           x[0] * y[0] * y[1] -
           x[1] * (square(y[1]) + square(y[2])); // grad kzeta matrix 1,3
  gkz[3] = (1 + epssq) * ((1 - square(x[0])) * y[2] + x[0] * x[2] * y[0]) -
           x[1] * y[1] * y[2] -
           x[2] * (square(y[0]) + square(y[2])); // grad kzeta matrix 2,1
  gkz[4] = ((1 + epssq) * x[1] - y[1]) *
           (x[2] * y[0] - x[0] * y[2]); // grad kzeta matrix 2,2
  gkz[5] = (1 + epssq) * (-(1 - square(x[2])) * y[0] - x[0] * x[2] * y[2]) +
           x[1] * y[1] * y[0] +
           x[0] * (square(y[0]) + square(y[2])); // grad kzeta matrix 2,3
  gkz[6] = (1 + epssq) * (-(1 - square(x[0])) * y[1] - x[0] * x[1] * y[0]) +
           x[2] * y[2] * y[1] +
           x[1] * (square(y[0]) + square(y[1])); // grad kzeta matrix 3,1
  gkz[7] = (1 + epssq) * ((1 - square(x[1])) * y[0] + x[0] * x[1] * y[1]) -
           x[2] * y[2] * y[0] -
           x[0] * (square(y[0]) + square(y[1])); // grad kzeta matrix 3,2
  gkz[8] = ((1 + epssq) * x[2] - y[2]) *
           (x[0] * y[1] - x[1] * y[0]); // grad kzeta matrix 3,3

  for (Short j = 0; j < 9; ++j) {
    gkz[j] *= denom;
  }
}
//
// template <typename Matrix3by3>
// KOKKOS_INLINE_FUNCTION
// Real double_dot_sphere(const Matrix3by3& mat) {
//   Real result = 0;
//   for (Int i=0; i<3; ++i) {
//     for (Int j=i; j<3; ++i) {
//       const Int ij_idx = 3*i + j;
//       const Int ji_idx = 3*j + i;
//       result += (i==j ? 1 : 2) * mat[ij_idx] * mat[ji_idx];
//     }
//   }
//   return result;
// }

template <typename Matrix3by3, typename XType, typename YType>
KOKKOS_INLINE_FUNCTION void grad_ksigma_sphere(Matrix3by3 &gks, const XType &x,
                                        const YType &y, const Real eps = 0) {
  const Real epssq = square(eps);
  const Real denom =
      1.0 / (4 * constants::PI * square(1 - SphereGeometry::dot(x, y) + epssq));

  gks[0] =
      -2 * x[0] * y[0] - 2 * epssq * x[0] * y[0] + 2 * cube(x[0]) * y[0] +
      2 * epssq * cube(x[0]) * y[0] + square(y[0]) -
      square(square(x[0])) * square(y[0]) - x[1] * y[1] - epssq * x[1] * y[1] +
      2 * square(x[0]) * x[1] * y[1] + 2 * epssq * square(x[0]) * x[1] * y[1] +
      x[0] * x[1] * y[0] * y[1] - 2 * cube(x[0]) * x[1] * y[0] * y[1] +
      square(x[1]) * square(y[1]) - square(x[0]) * square(x[1]) * square(y[1]) -
      x[2] * y[2] - epssq * x[2] * y[2] + 2 * square(x[0]) * x[2] * y[2] +
      2 * epssq * square(x[0]) * x[2] * y[2] + x[0] * x[2] * y[0] * y[2] -
      2 * cube(x[0]) * x[2] * y[0] * y[2] + 2 * x[1] * x[2] * y[1] * y[2] -
      2 * square(x[0]) * x[1] * x[2] * y[1] * y[2] +
      square(x[2]) * square(y[2]) - square(x[0]) * square(x[2]) * square(y[2]);

  gks[1] = 2 * square(x[0]) * x[1] * y[0] +
           2 * epssq * square(x[0]) * x[1] * y[0] - x[0] * x[1] * square(y[0]) -
           cube(x[0]) * x[1] * square(y[0]) - x[0] * y[1] -
           epssq * x[0] * y[1] + 2 * x[0] * square(x[1]) * y[1] +
           2 * epssq * x[0] * square(x[1]) * y[1] + y[0] * y[1] -
           square(x[1]) * y[0] * y[1] -
           2 * square(x[0]) * square(x[1]) * y[0] * y[1] -
           x[0] * cube(x[1]) * square(y[1]) + 2 * x[0] * x[1] * x[2] * y[2] +
           2 * epssq * x[0] * x[1] * x[2] * y[2] - x[1] * x[2] * y[0] * y[2] -
           2 * square(x[0]) * x[1] * x[2] * y[0] * y[2] -
           2 * x[0] * square(x[1]) * x[2] * y[1] * y[2] -
           x[0] * x[1] * square(x[2]) * square(y[2]);

  gks[2] = 2 * square(x[0]) * x[2] * y[0] +
           2 * epssq * square(x[0]) * x[2] * y[0] - x[0] * x[2] * square(y[0]) -
           cube(x[0]) * x[2] * square(y[0]) + 2 * x[0] * x[1] * x[2] * y[1] +
           2 * epssq * x[0] * x[1] * x[2] * y[1] - x[1] * x[2] * y[0] * y[1] -
           2 * square(x[0]) * x[1] * x[2] * y[0] * y[1] -
           x[0] * square(x[1]) * x[2] * square(y[1]) - x[0] * y[2] -
           epssq * x[0] * y[2] + 2 * x[0] * square(x[2]) * y[2] +
           2 * epssq * x[0] * square(x[2]) * y[2] + y[0] * y[2] -
           square(x[2]) * y[0] * y[2] -
           2 * square(x[0]) * square(x[2]) * y[0] * y[2] -
           2 * x[0] * x[1] * square(x[2]) * y[1] * y[2] -
           x[0] * cube(x[2]) * square(y[2]);

  gks[3] =
      -x[1] * y[0] - epssq * x[1] * y[0] + 2 * square(x[0]) * x[1] * y[0] +
      2 * epssq * square(x[0]) * x[1] * y[0] -
      cube(x[0]) * x[1] * square(y[0]) + 2 * x[0] * square(x[1]) * y[1] +
      2 * epssq * x[0] * square(x[1]) * y[1] + y[0] * y[1] -
      square(x[0]) * y[0] * y[1] -
      2 * square(x[0]) * square(x[1]) * y[0] * y[1] -
      x[0] * x[1] * square(y[1]) - x[0] * cube(x[1]) * square(y[1]) +
      2 * x[0] * x[1] * x[2] * y[2] + 2 * epssq * x[0] * x[1] * x[2] * y[2] -
      2 * square(x[0]) * x[1] * x[2] * y[0] * y[2] - x[0] * x[2] * y[1] * y[2] -
      2 * x[0] * square(x[1]) * x[2] * y[1] * y[2] -
      x[0] * x[1] * square(x[2]) * square(y[2]);

  gks[4] =
      -x[0] * y[0] - epssq * x[0] * y[0] + 2 * x[0] * square(x[1]) * y[0] +
      2 * epssq * x[0] * square(x[1]) * y[0] + square(x[0]) * square(y[0]) -
      square(x[0]) * square(x[1]) * square(y[0]) - 2 * x[1] * y[1] -
      2 * epssq * x[1] * y[1] + 2 * cube(x[1]) * y[1] +
      2 * epssq * cube(x[1]) * y[1] + x[0] * x[1] * y[0] * y[1] -
      2 * x[0] * cube(x[1]) * y[0] * y[1] + square(y[1]) -
      square(square(x[1])) * square(y[1]) - x[2] * y[2] - epssq * x[2] * y[2] +
      2 * square(x[1]) * x[2] * y[2] + 2 * epssq * square(x[1]) * x[2] * y[2] +
      2 * x[0] * x[2] * y[0] * y[2] -
      2 * x[0] * square(x[1]) * x[2] * y[0] * y[2] + x[1] * x[2] * y[1] * y[2] -
      2 * cube(x[1]) * x[2] * y[1] * y[2] + square(x[2]) * square(y[2]) -
      square(x[1]) * square(x[2]) * square(y[2]);

  gks[5] =
      2 * x[0] * x[1] * x[2] * y[0] + 2 * epssq * x[0] * x[1] * x[2] * y[0] -
      square(x[0]) * x[1] * x[2] * square(y[0]) +
      2 * square(x[1]) * x[2] * y[1] + 2 * epssq * square(x[1]) * x[2] * y[1] -
      x[0] * x[2] * y[0] * y[1] - 2 * x[0] * square(x[1]) * x[2] * y[0] * y[1] -
      x[1] * x[2] * square(y[1]) - cube(x[1]) * x[2] * square(y[1]) -
      x[1] * y[2] - epssq * x[1] * y[2] + 2 * x[1] * square(x[2]) * y[2] +
      2 * epssq * x[1] * square(x[2]) * y[2] -
      2 * x[0] * x[1] * square(x[2]) * y[0] * y[2] + y[1] * y[2] -
      square(x[2]) * y[1] * y[2] -
      2 * square(x[1]) * square(x[2]) * y[1] * y[2] -
      x[1] * cube(x[2]) * square(y[2]);

  gks[6] =
      -x[2] * y[0] - epssq * x[2] * y[0] + 2 * square(x[0]) * x[2] * y[0] +
      2 * epssq * square(x[0]) * x[2] * y[0] -
      cube(x[0]) * x[2] * square(y[0]) + 2 * x[0] * x[1] * x[2] * y[1] +
      2 * epssq * x[0] * x[1] * x[2] * y[1] -
      2 * square(x[0]) * x[1] * x[2] * y[0] * y[1] -
      x[0] * square(x[1]) * x[2] * square(y[1]) +
      2 * x[0] * square(x[2]) * y[2] + 2 * epssq * x[0] * square(x[2]) * y[2] +
      y[0] * y[2] - square(x[0]) * y[0] * y[2] -
      2 * square(x[0]) * square(x[2]) * y[0] * y[2] -
      x[0] * x[1] * y[1] * y[2] - 2 * x[0] * x[1] * square(x[2]) * y[1] * y[2] -
      x[0] * x[2] * square(y[2]) - x[0] * cube(x[2]) * square(y[2]);

  gks[7] = 2 * x[0] * x[1] * x[2] * y[0] +
           2 * epssq * x[0] * x[1] * x[2] * y[0] -
           square(x[0]) * x[1] * x[2] * square(y[0]) - x[2] * y[1] -
           epssq * x[2] * y[1] + 2 * square(x[1]) * x[2] * y[1] +
           2 * epssq * square(x[1]) * x[2] * y[1] -
           2 * x[0] * square(x[1]) * x[2] * y[0] * y[1] -
           cube(x[1]) * x[2] * square(y[1]) + 2 * x[1] * square(x[2]) * y[2] +
           2 * epssq * x[1] * square(x[2]) * y[2] - x[0] * x[1] * y[0] * y[2] -
           2 * x[0] * x[1] * square(x[2]) * y[0] * y[2] + y[1] * y[2] -
           square(x[1]) * y[1] * y[2] -
           2 * square(x[1]) * square(x[2]) * y[1] * y[2] -
           x[1] * x[2] * square(y[2]) - x[1] * cube(x[2]) * square(y[2]);

  gks[8] =
      -x[0] * y[0] - epssq * x[0] * y[0] + 2 * x[0] * square(x[2]) * y[0] +
      2 * epssq * x[0] * square(x[2]) * y[0] + square(x[0]) * square(y[0]) -
      square(x[0]) * square(x[2]) * square(y[0]) - x[1] * y[1] -
      epssq * x[1] * y[1] + 2 * x[1] * square(x[2]) * y[1] +
      2 * epssq * x[1] * square(x[2]) * y[1] + 2 * x[0] * x[1] * y[0] * y[1] -
      2 * x[0] * x[1] * square(x[2]) * y[0] * y[1] +
      square(x[1]) * square(y[1]) - square(x[1]) * square(x[2]) * square(y[1]) -
      2 * x[2] * y[2] - 2 * epssq * x[2] * y[2] + 2 * cube(x[2]) * y[2] +
      2 * epssq * cube(x[2]) * y[2] + x[0] * x[2] * y[0] * y[2] -
      2 * x[0] * cube(x[2]) * y[0] * y[2] + x[1] * x[2] * y[1] * y[2] -
      2 * x[1] * cube(x[2]) * y[1] * y[2] + square(y[2]) -
      square(square(x[2])) * square(y[2]);

  for (Short j = 0; j < 9; ++j) {
    gks[j] *= denom;
  }
}

} // namespace impl


/**  This interface allows the compiler to select the appropriate velocity kernel
  functions.   It must be specialized for each geometry type.
*/
template <typename Geo>
struct SWEVelocity {
  template <typename UType, typename XType, typename YType>
  KOKKOS_INLINE_FUNCTION
  static void kzeta(UType& u, const XType& x, const YType& y, const Real vort_y, const Real area_y, const Real eps = 0) {}

  template <typename UType, typename XType, typename YType>
  KOKKOS_INLINE_FUNCTION
  static void ksigma(UType& u, const XType& x, const YType& y, const Real div_y, const Real area_y, const Real eps=0) {}

  template <typename MatrixType, typename XType, typename YType>
  KOKKOS_INLINE_FUNCTION
  static void grad_kzeta(MatrixType& gkz, const XType& x, const YType& y, const Real eps = 0) {}

  template <typename MatrixType, typename XType, typename YType>
  KOKKOS_INLINE_FUNCTION
  static void grad_ksigma(MatrixType& gks, const XType& x, const YType& y, const Real eps = 0) {}
};


template <>
struct SWEVelocity<PlaneGeometry> {
  /**  Biot-Savart kernel for planar problems.

  @param [in/out] u velocity contribution from vorticity
  @param [in] x target location
  @param [in] y source location
  @param [in] vort_y source vorticity
  @param [in] area_y source area
  @param [in] eps regularization parameter
*/
  template <typename UType, typename XType, typename YType>
  KOKKOS_INLINE_FUNCTION
  static void kzeta(UType& u, const XType& x, const YType& y, const Real vort_y, const Real area_y, const Real eps = 0) {
    LPM_KERNEL_ASSERT(eps >= 0);
    return impl::kzeta_plane(u, x, y, vort_y, area_y, eps);
  }

  /**  Scalar potential velocity kernel for planar problems.

  @param [in/out] u velocity contribution from vorticity
  @param [in] x target location
  @param [in] y source location
  @param [in] div_y source vorticity
  @param [in] area_y source area
  @param [in] eps regularization parameter
*/
template <typename UType, typename XType, typename YType>
KOKKOS_INLINE_FUNCTION
static void ksigma(UType &u, const XType &x, const YType &y, const Real div_y, const Real area_y, const Real eps = 0) {
  LPM_KERNEL_ASSERT(eps >= 0);
  return impl::ksigma_plane(u, x, y, div_y, area_y, eps);
}

/** Tensor gradient of the Biot-Savart kernel for the plane.

  @param [out] gkz  gradient matrix
  @param [in] x target location
  @param [in] y source location
  @param [in] eps regularization parameter
*/
template <typename MatrixType, typename XType, typename YType>
KOKKOS_INLINE_FUNCTION
static void grad_kzeta(MatrixType& gkz, const XType& x, const YType& y, const Real eps=0) {
  LPM_KERNEL_ASSERT(eps >= 0);
  return impl::grad_kzeta_plane(gkz, x, y, eps);
}

/** Tensor gradient of the scalar potential velocity kernel for the plane.

  @param [out] gkz  gradient matrix
  @param [in] x target location
  @param [in] y source location
  @param [in] eps regularization parameter
*/
template <typename MatrixType, typename XType, typename YType>
KOKKOS_INLINE_FUNCTION
static void grad_ksigma(MatrixType& gks, const XType& x, const YType& y, const Real eps = 0) {
  return impl::grad_ksigma_plane(gks, x, y, eps);
}

};


template<>
struct SWEVelocity<SphereGeometry> {
  /**  Biot-Savart kernel for spherical problems.

  @param [in/out] u velocity contribution from vorticity
  @param [in] x target location
  @param [in] y source location
  @param [in] vort_y source vorticity
  @param [in] area_y source area
  @param [in] eps regularization parameter
*/
template <typename UType, typename XType, typename YType>
KOKKOS_INLINE_FUNCTION
static void kzeta(UType &u, const XType &x, const YType &y, const Real vort_y, const Real area_y, const Real eps = 0) {
  LPM_KERNEL_ASSERT(eps >= 0);
  return impl::kzeta_sphere(u, x, y, vort_y, area_y, eps);
}

/**  Scalar potential velocity kernel for spherical problems.

  @param [in/out] u velocity contribution from vorticity
  @param [in] x target location
  @param [in] y source location
  @param [in] div_y source vorticity
  @param [in] area_y source area
  @param [in] eps regularization parameter
*/
template <typename UType, typename XType, typename YType>
KOKKOS_INLINE_FUNCTION
static void ksigma(UType &u, const XType &x, const YType &y, const Real div_y, const Real area_y, const Real eps = 0) {
  LPM_KERNEL_ASSERT(eps >= 0);
  return impl::ksigma_sphere(u, x, y, div_y, area_y, eps);
}

/** Tensor gradient of the Biot-Savart kernel for the sphere.

  @param [out] gkz  gradient matrix
  @param [in] x target location
  @param [in] y source location
  @param [in] eps regularization parameter
*/
template <typename MatrixType, typename XType, typename YType>
KOKKOS_INLINE_FUNCTION
static void grad_kzeta(MatrixType& gkz, const XType& x, const YType& y, const Real eps=0) {
  LPM_KERNEL_ASSERT(eps >= 0);
  return impl::grad_kzeta_sphere(gkz, x, y, eps);
}

/** Tensor gradient of the scalar potential velocity kernel for the sphere.

  @param [out] gkz  gradient matrix
  @param [in] x target location
  @param [in] y source location
  @param [in] eps regularization parameter
*/
template <typename MatrixType, typename XType, typename YType>
KOKKOS_INLINE_FUNCTION
static void grad_ksigma(MatrixType& gks, const XType& x, const YType& y, const Real eps = 0) {
  return impl::grad_ksigma_sphere(gks, x, y, eps);
}

};



} // namespace Lpm
#endif
