#ifndef LPM_SWE_KERNELS_HPP
#define LPM_SWE_KERNELS_HPP

#include "LpmConfig.h"
#include "lpm_geometry.hpp"
#include "util/lpm_math.hpp"

namespace Lpm {

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
    u[j] += uloc[j] * strength;
  }
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
    u[j] += uloc[j] * strength;
  }
}

template <typename Compressed3by3, typename XType, typename YType>
KOKKOS_INLINE_FUNCTION void grad_kzeta(Compressed3by3 &gkz, const XType &x,
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

template <typename Compressed3by3, typename XType, typename YType>
KOKKOS_INLINE_FUNCTION void grad_ksigma(Compressed3by3 &gks, const XType &x,
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

} // namespace Lpm

#endif
