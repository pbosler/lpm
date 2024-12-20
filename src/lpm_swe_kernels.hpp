#ifndef LPM_SWE_KERNELS_HPP
#define LPM_SWE_KERNELS_HPP

#include "LpmConfig.h"
#include "lpm_assert.hpp"
#include "lpm_coriolis.hpp"
#include "lpm_geometry.hpp"
#include "lpm_pse.hpp"
#include "util/lpm_math.hpp"

namespace Lpm {

/** @brief Evaluates the Biot-Savart kernel in the plane.

    @param [out] u velocity contribution from source y on target x
    @param [in] x target point
    @param [in] y source point
    @param [in] eps velocity smoothing parameter
*/
template <typename UType, typename XType, typename YType>
KOKKOS_INLINE_FUNCTION void kzeta_plane(
  UType& u, /// [out] kernel output
  XType& x, /// [in] target coordinates
  YType& y, /// [in] source coordinates
  Real& eps /// [in] regularization parameter
) {
  const Real denom = 2 * constants::PI * (square(PlaneGeometry::distance(x, y)) + square(eps));
  u[0] = -(x[1] - y[1]) / denom;
  u[1] =  (x[0] - y[0]) / denom;
}

/** @brief Evaluates the velocity potential kernel in the plane.

    @param [out] u velocity contribution from source y on target x
    @param [in] x target point
    @param [in] y source point
    @param [in] eps velocity smoothing parameter
*/
template <typename UType, typename XType, typename YType>
KOKKOS_INLINE_FUNCTION void ksigma_plane(
  UType& u, /// [out] kernel output
  XType& x, /// [in] target coordinates
  YType& y, /// [in] source coordinates
  Real& eps /// [in] regularization parameter
) {
  const Real denom = 2 * constants::PI * (square(PlaneGeometry::distance(x, y)) + square(eps));
  u[0] = -(x[1] - y[1]) / denom;
  u[1] =  (x[0] - y[0]) / denom;
}

/** @brief Biot-Savart kernel for the sphere

    @param [out] u velocity contribution from source y on target x
    @param [in] x target point
    @param [in] y source point
    @param [in] vort_y relative vorticity at y
    @param [in] area_y area weight of panel y
    @param [in] eps kernel smoothing parameter
*/
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

/** @brief gradient of the potential kernel for the sphere

    @param [out] u velocity contribution from source y on target x
    @param [in] x target point
    @param [in] y source point
    @param [in] div_y divergence of velocity at y
    @param [in] area_y area weight of panel y
    @param [in] eps kernel smoothing parameter
*/
template <typename UType, typename XType, typename YType>
KOKKOS_INLINE_FUNCTION void
ksigma_sphere(UType &u, const XType &x, const YType &y, const Real div_y,
              const Real area_y, const Real eps = 0) {
  const Real denom =
      4 * constants::PI * (1 - SphereGeometry::dot(x, y) + square(eps));
  const Real strength = -div_y * area_y / denom;
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

/** @brief Gradient of the Biot-Savart kernel (a 2-tensor)

    @param [out] gkz contribution from source y on target x
    @param [in] x target point
    @param [in] y source point
    @param [in] eps kernel smoothing parameter
*/
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


/** @brief Computes the double dot product sum, given
  the gradient of velocity tensor

    @param [in] mat a 1d array of size 9 containing the packed (row-major)
                    entries of the velocity gradient matrix, @f$ \nabla \vec u  @f$
    @return @f$ \nabla \vec{u} : \nabla \vec{u}^T @f$
*/
template <typename Compressed3by3>
KOKKOS_INLINE_FUNCTION
Real double_dot(const Compressed3by3& mat) {
  Real result = 0;
  for (Int i=0; i<3; ++i) {
    for (Int j=i; j<3; ++i) {
      const Int ij_idx = 3*i + j;
      const Int ji_idx = 3*j + i;
      result += /*(i==j ? 1 : 2) * */ mat[ij_idx] * mat[ji_idx];
    }
  }
  return result;
}

/** @brief Gradient of the velocity potential kernel (a 2-tensor)

    @param [out] gks contribution from source y on target x
    @param [in] x target point
    @param [in] y source point
    @param [in] eps kernel smoothing parameter
*/
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

/** Computes the direct sums for spherical velocity in a shallow water problem

  The output tuple is packed with the following entries:

  index 0 : u
  index 1 : v
  index 2 : w
  index 3 : du/dx
  index 4 : du/dy
  index 5 : du/dz
  index 6 : dv/dx
  index 7 : dv/dy
  index 8 : dv/dz
  index 9 : dw/dx
  index 10: dw/dy
  index 11: dw/dz

  @param [in] tgt_x target point coordinates
  @param [in] src_y source point coordinates
  @param [in] src_zeta vorticity at source point
  @param [in] src_sigma divergence at source point
  @param [in] src_area area weight of source point
  @param [in] eps kernel smoothing parameter
  @return packed contributions of source y on target x
*/
template <typename XType, typename YType>
KOKKOS_INLINE_FUNCTION
Kokkos::Tuple<Real,12> sphere_swe_velocity_sums(const XType& tgt_x,
  const YType& src_y,
  const Real& src_zeta,
  const Real& src_sigma,
  const Real& src_area,
  const Real& eps) {
  Kokkos::Tuple<Real,12> result;
  // compute the velocity contribution from vorticity interaction
  // between target particle x and source particle y
  Real velz[3];
  kzeta_sphere( velz, tgt_x, src_y, src_zeta, src_area, eps);
  // sum in the velocity contribution from dilatation interaction
  // between target particle x and source particle y
  Real vels[3];
  ksigma_sphere(vels, tgt_x, src_y, src_sigma,src_area, eps);
  for (int k=0; k<3; ++k) {
    result[k] = velz[k] + vels[k];
  }
  // compute the solenoidal velocity gradient contributions
  Real gkz[9];
  grad_kzeta(gkz, tgt_x, src_y, eps);
  // compute the irrotational velocity gradient contributions
  Real gks[9];
  grad_ksigma(gks, tgt_x, src_y, eps);
  const Real rot_str = -src_zeta * src_area;
  const Real pot_str = -src_sigma * src_area;
  for (int k=0; k<9; ++k) {
    result[3+k] += gkz[k]*rot_str + gks[k]*pot_str;
  }
  return result;
}

/** Evaluates the RHS terms for the planar shallow water equations
  using Particle Strength Exchange (PSE) to compute the Laplacian
  of the fluid surface.

  Each call to this function computes 1 pairwise interaction, the
  contributions of source particle y to target particle x.

  Results are packed into a 9-tuple:
  index 0: u
  index 1: v
  index 2: du/dx
  index 3: du/dy
  index 4: dv/dx
  index 5: dv/dy
  index 6: laplacian(sfc) from PSE
  index 7: stream function
  index 8: potential function

  @param [in] tgt_x target point coordinates
  @param [in] src_y source point coordinates
  @param [in] src_zeta vorticity at source point
  @param [in] src_sigma divergence at source point
  @param [in] src_area area weight of source point
  @param [in] src_s surface height at source point
  @param [in] tgt_s surface height at target point
  @param [in] eps velocity kernel smoothing parameter
  @param [in] pse_eps PSE kernel width parameter
  @return packed contributions of source y on target x
*/
template <typename XType, typename YType>
KOKKOS_INLINE_FUNCTION
Kokkos::Tuple<Real, 9> planar_swe_sums_rhs_pse(
  const XType& tgt_x, /// coordinates of target point
  const YType& src_y, /// coordinates of source point
  const Real& src_zeta, /// source vorticity
  const Real& src_sigma, /// source divergence
  const Real& src_area, /// source area
  const Real& src_s, /// source surface height
  const Real& tgt_s, /// target surface height
  const Real& eps, /// velocity kernel regularization parameter
  const Real& pse_eps /// PSE kernel width parameter
){
  using pse_type = pse::BivariateOrder8;
  using value_type = Kokkos::Tuple<Real,9>;
  value_type result;

  const Real x_minus_y[2] = {tgt_x[0] - src_y[0], tgt_x[1] - src_y[1]};
  const Real epssq = square(eps);
  const Real rsq = PlaneGeometry::norm2(x_minus_y);
  const Real denom  = 2*constants::PI * (rsq + epssq);
  const Real denom2 = constants::PI * square(rsq +  epssq);
  const Real rot_str = src_zeta * src_area;
  const Real pot_str = src_sigma * src_area;
  const Real greens = -log(rsq + epssq) / (4 * constants::PI);
  const Real x_mix_y = x_minus_y[0] * x_minus_y[1];

  // u
  result[0] = (-x_minus_y[1]*rot_str + x_minus_y[0]*pot_str) / denom;
  // v
  result[1] = ( x_minus_y[0]*rot_str + x_minus_y[1]*pot_str) / denom;

  // du/dx
  result[2] = ( x_mix_y * rot_str - square(x_minus_y[0]) * pot_str) / denom2 + pot_str / denom;
  // du/dy
  result[3] = ( square(x_minus_y[1]) * rot_str - x_mix_y * pot_str) / denom2 - rot_str / denom;
  // dv/dx
  result[4] = (-square(x_minus_y[0]) * rot_str - x_mix_y * pot_str) / denom2 + rot_str / denom;
  // dv/dy
  result[5] = (-x_mix_y * rot_str - square(x_minus_y[1]) * pot_str) / denom2 + pot_str / denom;


  // lap(s)
  const Real pse_input = pse_type::kernel_input(tgt_x, src_y, pse_eps);
  const Real pse_val = pse_type::laplacian(pse_input);
  result[6] = (src_s - tgt_s) * src_area * pse_val / square(pse_eps);
  result[7] = greens * rot_str;
  result[8] = greens * pot_str;

  return result;
}

/** Evaluates the RHS terms for the spherical shallow water equations
  using Particle Strength Exchange (PSE) to compute the Laplacian
  of the fluid surface.

  Each call to this function computes 1 pairwise interaction, the
  contributions of source particle y to target particle x.

  Results are packed into a 13-tuple:

  index 0 : u
  index 1 : v
  index 2 : w
  index 3 : du/dx
  index 4 : du/dy
  index 5 : du/dz
  index 6 : dv/dx
  index 7 : dv/dy
  index 8 : dv/dz
  index 9 : dw/dx
  index 10: dw/dy
  index 11: dw/dz
  index 12: laplacian(sfc)

  @param [in] tgt_x target point coordinates
  @param [in] src_y source point coordinates
  @param [in] src_zeta vorticity at source point
  @param [in] src_sigma divergence at source point
  @param [in] src_area area weight of source point
  @param [in] src_s surface height at source point
  @param [in] tgt_s surface height at target point
  @param [in] eps velocity kernel smoothing parameter
  @param [in] pse_eps PSE kernel width parameter
  @return packed contributions of source y on target x
*/
template <typename XType, typename YType>
KOKKOS_INLINE_FUNCTION
Kokkos::Tuple<Real,13> sphere_swe_sums_rhs_pse(
  const XType& tgt_x,
  const YType& src_y,
  const Real& src_zeta,
  const Real& src_sigma,
  const Real& src_area,
  const Real& src_s,
  const Real& tgt_s,
  const Real& eps,
  const Real& pse_eps
  )
{
  using pse_type = pse::BivariateOrder8;
  Kokkos::Tuple<Real,13> result;

  // compute the velocity contribution from vorticity interaction
  // between target particle x and source particle y
  Real vel[3];
  kzeta_sphere( vel, tgt_x, src_y, src_zeta, src_area, eps);
  // sum in the velocity contribution from dilatation interaction
  // between target particle x and source particle y
  ksigma_sphere(vel, tgt_x, src_y, src_sigma,src_area, eps);
  // compute the solenoidal velocity gradient contributions
  Real gkz[9];
  grad_kzeta(gkz, tgt_x, src_y, eps);
  // compute the irrotational velocity gradient contributions
  Real gks[9];
  grad_ksigma(gks, tgt_x, src_y, eps);
  const Real rot_str = src_zeta * src_area;
  const Real pot_str = src_sigma * src_area;
  for (int k=0; k<9; ++k) {
    result[3+k] += gkz[k]*rot_str + gks[k]*pot_str;
  }
  if (SphereGeometry::dot(tgt_x, src_y) > 0) {
    const Real pse_input = pse_type::kernel_input(tgt_x, src_y, pse_eps);
    const Real pse_val = pse_type::laplacian(pse_input);
    result[12] = (src_s - tgt_s) * src_area * pse_val / square(pse_eps);
  }
  else {
    result[12] = 0;
  }
  return result;
}

/** @brief Kokkos Reducer concept for planar SWE direct summation,
  with Particle Strength Exchange for the Laplacian.

  This functor will be called from a Kokkos::parallel_reduce kernel.
*/
struct PlanarSwePseDirectSumReducer {
  using crd_view = PlaneGeometry::crd_view_type;
  using value_type = Kokkos::Tuple<Real,9>;
  crd_view tgt_x; // [in] coordinates of target points
  scalar_view_type tgt_sfc; // [in] surface height of target points
  Index i; // [in] index of target point in target views
  crd_view src_y; // [in] coordinates of source points
  bool collocated_src_tgt; // [in] true if src points and target points are collocated
  scalar_view_type src_zeta; // [in] relative vorticity at source points
  scalar_view_type src_sigma; // [in] divergence at source points
  scalar_view_type src_area; // [in] area of source point panels
  mask_view_type src_mask; // [in] divided panels are masked
  scalar_view_type src_sfc; // [in] surface height at target points
  Real eps; // velocity kernel smoothing parameter
  Real pse_eps; // PSE kernel width parameter


  KOKKOS_INLINE_FUNCTION
  PlanarSwePseDirectSumReducer(const crd_view tx,
    const scalar_view_type tsfc,
    const Index idx,
    const crd_view sy,
    const bool collocated,
    const scalar_view_type szeta,
    const scalar_view_type ssigma,
    const scalar_view_type sarea,
    const mask_view_type smask,
    const scalar_view_type ssfc,
    const Real eps,
    const Real pse_eps) :
    tgt_x(tx),
    tgt_sfc(tsfc),
    i(idx),
    src_y(sy),
    collocated_src_tgt(collocated),
    src_zeta(szeta),
    src_sigma(ssigma),
    src_area(sarea),
    src_mask(smask),
    src_sfc(ssfc),
    eps(eps),
    pse_eps(pse_eps) {}


  KOKKOS_INLINE_FUNCTION
  void operator() (const Index& j, value_type& r) const {
    const bool compute_ij = (!src_mask(j) and (!collocated_src_tgt or i != j));
    if (compute_ij) {
      const auto xcrd = Kokkos::subview(tgt_x, i, Kokkos::ALL);
      const auto ycrd = Kokkos::subview(src_y, j, Kokkos::ALL);
      const value_type local_result =
        planar_swe_sums_rhs_pse(xcrd, ycrd, src_zeta(j), src_sigma(j),
          src_area(j), src_sfc(j), tgt_sfc(i), eps, pse_eps);
      r += local_result;
    }
  }
};

/** @brief Kokkos reducer concept for the spherical shallow water equations,
  velocity sums only (no PSE)

  This functor will be called from a Kokkos::parallel_reduce kernel.
*/
struct SphereSweDirectSumReducer {
  using crd_view = SphereGeometry::crd_view_type;
  using value_type = Kokkos::Tuple<Real,12>;
  crd_view tgt_x;
  Index i;
  crd_view src_y;
  bool collocated_src_tgt;
  scalar_view_type src_zeta;
  scalar_view_type src_sigma;
  scalar_view_type src_area;
  mask_view_type src_mask;
  Real eps;

  SphereSweDirectSumReducer(const crd_view& tx, const Index tgt_idx,
    const crd_view& sy, const bool colloc, const scalar_view_type& sz,
    const scalar_view_type& ss, const scalar_view_type& sa,
    const mask_view_type m, const Real eps) :
    tgt_x(tx),
    i(tgt_idx),
    src_y(sy),
    collocated_src_tgt(colloc),
    src_zeta(sz),
    src_sigma(ss),
    src_area(sa),
    src_mask(m),
    eps(eps) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index& j, value_type& r) const {
    if (!src_mask(j)) {
      if (!collocated_src_tgt or i != j ) {
        const auto xcrd = Kokkos::subview(tgt_x, i, Kokkos::ALL);
        const auto ycrd = Kokkos::subview(src_y, j, Kokkos::ALL);
        const value_type local_result =
          sphere_swe_velocity_sums(xcrd, ycrd, src_zeta(j), src_sigma(j),
            src_area(j), eps);
        r += local_result;
      }
    }
  }
};

/** @brief Direct sum functor for Planar SWE vertices (passive particles).

  Dispatches one thread team per target point.  Each thread team
  performs a direct sum reduction.
*/
struct PlanarSWEVertexSums {
  using vec_view = PlaneGeometry::vec_view_type;
  using crd_view = PlaneGeometry::crd_view_type;

  vec_view vert_u; /// [out] velocity at vertices
  scalar_view_type vert_ddot; /// [out] double dot product at vertices
  scalar_view_type vert_du1dx1; /// [out]
  scalar_view_type vert_du1dx2; /// [out]
  scalar_view_type vert_du2dx1; /// [out]
  scalar_view_type vert_du2dx2; /// [out]
  scalar_view_type vert_laps; /// [out] surface Laplacian at vertices
  scalar_view_type vert_psi; /// [out] stream function at vertices
  scalar_view_type vert_phi; /// [out] potential function at vertices
  crd_view vert_x; /// [in] vertex coordinates (targets)
  scalar_view_type vert_s;; /// [in] surface height at vertices
  crd_view face_y; /// [in] face coordinates (sources)
  scalar_view_type face_zeta; /// [in] vorticity at faces
  scalar_view_type face_sigma; /// [in] divergence at faces
  scalar_view_type face_area; /// [in] area of faces
  mask_view_type face_mask; /// [in] mask to exclude divided faces
  scalar_view_type face_s; /// [in] surface height at faces
  Real eps; /// [in] velocity kernel smoothing parameter
  Real pse_eps; /// [in] PSE kernel width parameter
  Index nfaces; /// [in] total number of faces (including divided faces)
  bool do_velocity; /// [in] if true, overwrite velocity

  PlanarSWEVertexSums(vec_view& vu, scalar_view_type& vdd,
    scalar_view_type du1dx1,
    scalar_view_type du1dx2,
    scalar_view_type du2dx1,
    scalar_view_type du2dx2,
    scalar_view_type& vls,
    scalar_view_type& vpsi, scalar_view_type& vphi,
    const crd_view vx, const scalar_view_type vsfc, const crd_view fy,
    const scalar_view_type fzeta, const scalar_view_type fsig,
    const scalar_view_type farea, const mask_view_type fmask,
    const scalar_view_type fsfc, const Real eps, const Real pse_eps,
    const Index nfaces, const bool do_velocity = true) :
    vert_u(vu),
    vert_ddot(vdd),
    vert_du1dx1(du1dx1),
    vert_du1dx2(du1dx2),
    vert_du2dx1(du2dx1),
    vert_du2dx2(du2dx2),
    vert_laps(vls),
    vert_psi(vpsi),
    vert_phi(vphi),
    vert_x(vx),
    vert_s(vsfc),
    face_y(fy),
    face_zeta(fzeta),
    face_sigma(fsig),
    face_area(farea),
    face_mask(fmask),
    face_s(fsfc),
    eps(eps),
    pse_eps(pse_eps),
    nfaces(nfaces),
    do_velocity(do_velocity) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const member_type& thread_team) const {
    const Index i = thread_team.league_rank();

    // perform packed reduction, dispatching 1 thread team per target
    Kokkos::Tuple<Real,9> sums;
    constexpr bool collocated = false;
    Kokkos::parallel_reduce(Kokkos::TeamVectorRange(thread_team, nfaces),
      PlanarSwePseDirectSumReducer(vert_x, vert_s, i, face_y, collocated,
        face_zeta, face_sigma, face_area, face_mask, face_s,
        eps, pse_eps), sums);

    // unpack reduction results
    if (do_velocity) {
      vert_u(i,0) = sums[0];
      vert_u(i,1) = sums[1];
    }

//
    vert_ddot(i) = square(sums[2]) + 2*sums[3]*sums[4] + square(sums[5]);
    vert_du1dx1(i) = sums[2];
    vert_du1dx2(i) = sums[3];
    vert_du2dx1(i) = sums[4];
    vert_du2dx2(i) = sums[5];
    vert_laps(i) = sums[6];
    vert_psi(i) = sums[7];
    vert_phi(i) = sums[8];
  }
};


/** @brief Direct sum functor for spherical SWE vertices (passive particles).

  Dispatches one thread team per target point.  Each thread team
  performs a direct sum reduction.
*/
struct SphereVertexSums {
  using vec_view = SphereGeometry::vec_view_type;
  using crd_view = SphereGeometry::crd_view_type;

  vec_view vert_u;
  scalar_view_type vert_ddot;
  crd_view vert_x;
  crd_view face_y;
  scalar_view_type face_zeta;
  scalar_view_type face_sigma;
  scalar_view_type face_area;
  mask_view_type face_mask;
  Real eps;
  Index nfaces;
  bool do_velocity;

  SphereVertexSums(vec_view& vu, scalar_view_type& vdd,
    const crd_view& vx, const crd_view& fy,
    const scalar_view_type& fz, const scalar_view_type& fs,
    const scalar_view_type& fa, const mask_view_type& fm,
    const Real eps, const Index nf, const bool do_velocity) :
    vert_u(vu),
    vert_ddot(vdd),
    vert_x(vx),
    face_y(fy),
    face_zeta(fz),
    face_sigma(fs),
    face_area(fa),
    face_mask(fm),
    eps(eps),
    nfaces(nf),
    do_velocity(do_velocity) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const member_type& thread_team) const {
    const Index i = thread_team.league_rank();

    Kokkos::Tuple<Real, 12> sums;
    constexpr bool collocated = false;
    Kokkos::parallel_reduce(Kokkos::TeamVectorRange(thread_team, nfaces),
      SphereSweDirectSumReducer(vert_x, i, face_y, collocated, face_zeta,
        face_sigma, face_area, face_mask, eps), sums);

    if (do_velocity) {
      for (int k = 0; k<3; ++k) {
        vert_u(i,k) = sums[k];
      }
    }
    vert_ddot(i) = 0;
    for (int ii=0; ii<3; ++ii) {
      for (int jj=0; jj<3; ++jj) {
        vert_ddot(i) += sums[3 + 3*ii+jj] * sums[3 + 3*jj+ii];
      }
    }
  }
};

/** @brief Direct sum functor for Planar SWE (active particles).

  Dispatches one thread team per target point.  Each thread team
  performs a direct sum reduction.
*/
struct PlanarSWEFaceSums {
  using vec_view = PlaneGeometry::vec_view_type;
  using crd_view = PlaneGeometry::crd_view_type;

  vec_view face_u; /// [out] velocity at faces (targets)
  scalar_view_type face_ddot; /// [out] double dot product at faces
  scalar_view_type face_du1dx1; /// [out]
  scalar_view_type face_du1dx2; /// [out]
  scalar_view_type face_du2dx1; /// [out]
  scalar_view_type face_du2dx2; /// [out]
  scalar_view_type face_laps; /// [out] surface Laplacian at faces
  scalar_view_type face_psi; /// [out] stream function at faces
  scalar_view_type face_phi; /// [out] potential function at faces
  crd_view face_xy; /// [in] coordinates of faces (sources and targets)
  scalar_view_type face_zeta; /// [in] vorticity at faces
  scalar_view_type face_sigma; /// [in] divergence at faces
  scalar_view_type face_area; /// [in] face area
  mask_view_type face_mask; /// [in] mask to exclude divided panels
  scalar_view_type face_s; /// [in] surface height at faces
  Real eps; /// [in] velocity kernel smoothing parameter
  Real pse_eps; /// [in] PSE kernel width parameter
  Index nfaces; /// [in] total number of faces (including divided faces)
  bool do_velocity; /// [in] if true, overwrite velocity

  PlanarSWEFaceSums(vec_view& fu, scalar_view_type& fdd,
    scalar_view_type du1dx1,
    scalar_view_type du1dx2,
    scalar_view_type du2dx1,
    scalar_view_type du2dx2,
    scalar_view_type& fls,
    scalar_view_type& fpsi, scalar_view_type& fphi,
    const crd_view fxy, const scalar_view_type fzeta, const scalar_view_type fsig,
    const scalar_view_type farea, const mask_view_type fmask, const scalar_view_type fsfc,
    const Real eps, const Real pse_eps, const Index nfaces,
    const bool do_velocity = true) :
    face_u(fu),
    face_ddot(fdd),
    face_du1dx1(du1dx1),
    face_du1dx2(du1dx2),
    face_du2dx1(du2dx1),
    face_du2dx2(du2dx2),
    face_laps(fls),
    face_psi(fpsi),
    face_phi(fphi),
    face_xy(fxy),
    face_zeta(fzeta),
    face_sigma(fsig),
    face_area(farea),
    face_mask(fmask),
    face_s(fsfc),
    eps(eps),
    pse_eps(pse_eps),
    nfaces(nfaces),
    do_velocity(do_velocity) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const member_type& thread_team) const {
    const Index i = thread_team.league_rank();

    // perform packed reduction, 1 thread team per target
    Kokkos::Tuple<Real,9> sums;
    const bool collocated = !(eps > 0);
    Kokkos::parallel_reduce(Kokkos::TeamVectorRange(thread_team, nfaces),
      PlanarSwePseDirectSumReducer(face_xy, face_s, i, face_xy, collocated,
        face_zeta, face_sigma, face_area, face_mask, face_s,
        eps, pse_eps), sums);

    // unpack reduction results
    if (do_velocity) {
      face_u(i,0) = sums[0];
      face_u(i,1) = sums[1];
    }

    face_ddot(i) = square(sums[2]) + 2*sums[3]*sums[4] + square(sums[5]);
    face_du1dx1(i) = sums[2];
    face_du1dx2(i) = sums[3];
    face_du2dx1(i) = sums[4];
    face_du2dx2(i) = sums[5];
    face_laps(i) = sums[6];
    face_psi(i) = sums[7];
    face_phi(i) = sums[8];
  }
};


/** @brief Direct sum functor for spherical SWE (active particles).

  Dispatches one thread team per target point.  Each thread team
  performs a direct sum reduction.
*/
struct SphereFaceSums {
  using vec_view = SphereGeometry::vec_view_type;
  using crd_view = SphereGeometry::crd_view_type;

  vec_view face_u;
  scalar_view_type face_ddot;
  crd_view face_xy;
  scalar_view_type face_zeta;
  scalar_view_type face_sigma;
  scalar_view_type face_area;
  mask_view_type face_mask;
  Real eps;
  Index nfaces;
  bool do_velocity;

  SphereFaceSums(vec_view& fu, scalar_view_type& fdd,
    const crd_view& fxy, const scalar_view_type& fz,
    const scalar_view_type& fs, const scalar_view_type& fa,
    const mask_view_type& fm, const Real eps,
    const Index nf, const bool do_velocity) :
    face_u(fu),
    face_ddot(fdd),
    face_xy(fxy),
    face_zeta(fz),
    face_sigma(fs),
    face_area(fa),
    face_mask(fm),
    eps(eps),
    nfaces(nf),
    do_velocity(do_velocity) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const member_type& thread_team) const {
    const Index i = thread_team.league_rank();

    Kokkos::Tuple<Real,12> sums;
    const bool collocated = FloatingPoint<Real>::zero(eps);
    Kokkos::parallel_reduce(Kokkos::TeamVectorRange(thread_team, nfaces),
      SphereSweDirectSumReducer(face_xy, i, face_xy, collocated, face_zeta,
        face_sigma, face_area, face_mask, eps), sums);
    if (do_velocity) {
      for (int k = 0; k<3; ++k) {
        face_u(i,k) = sums[k];
      }
    }
    face_ddot(i) = 0;
    for (int ii=0; ii<3; ++ii) {
      for (int jj=0; jj<3; ++jj) {
        face_ddot(i) += sums[3 + 3*ii+jj] * sums[3 + 3*jj+ii];
      }
    }
  }
};


/** @brief  RHS tendencies for the shallow water equations
  at passive particles (both planar and spherical problems)

  If U is the state vector, and the SWE are written as a dynamical
  system, dU/dt = F(U), this functor computes the RHS F, given U.

  This functor will be called from a Kokkos::parallel_for, with a range
  policy over all passive particles.
*/
template <typename Geo>
struct SWEVorticityDivergenceHeightTendencies {
  using crd_view = typename Geo::crd_view_type;
  using vec_view = typename Geo::vec_view_type;
  using coriolis_type = typename std::conditional<
    std::is_same<Geo, PlaneGeometry>::value, CoriolisBetaPlane,
      CoriolisSphere>::type;
  scalar_view_type dzeta; /// [out] vorticity tendency
  scalar_view_type dsigma; /// [out] divergence tendency
  scalar_view_type dh; /// [out] depth tendency
  crd_view x; /// [in] physical coordinates
  vec_view u; /// [in] velocity
  scalar_view_type zeta; /// [in] vorticity
  scalar_view_type sigma; /// [in] divergence
  scalar_view_type h; /// [in] depth
  scalar_view_type ddot; /// [in] double dot product
  scalar_view_type laps; /// [in] surface Laplacian
  coriolis_type coriolis; /// [in] Coriolis functor
  Real g; /// [in] gravity
  Real dt; /// [in] time step size

  SWEVorticityDivergenceHeightTendencies(scalar_view_type& dzeta,
    scalar_view_type& dsigma, scalar_view_type& dh,
    const crd_view& x, const vec_view& u,
    const scalar_view_type zeta, const scalar_view_type sigma,
    const scalar_view_type h, const scalar_view_type ddot,
    const scalar_view_type laps,
    const coriolis_type& cor, const Real g,
    const Real dt) :
    dzeta(dzeta),
    dsigma(dsigma),
    dh(dh),
    x(x),
    u(u),
    zeta(zeta),
    sigma(sigma),
    h(h),
    ddot(ddot),
    laps(laps),
    g(g),
    coriolis(cor),
    dt(dt) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index i) const {
    const auto xi = Kokkos::subview(x, i, Kokkos::ALL);
    const auto ui = Kokkos::subview(u, i, Kokkos::ALL);
    const Real f = coriolis.f(xi);
    dzeta(i) = (-coriolis.dfdt(ui) - (zeta(i) + f) * sigma(i))*dt;
    if constexpr (std::is_same<Geo, SphereGeometry>::value) {
      dsigma(i) = (f*zeta(i) + coriolis.grad_f_cross_u(xi,ui)
        - ddot(i) - g*laps(i) - SphereGeometry::norm2(ui))*dt;
    }
    else {
      dsigma(i) = (f*zeta(i) + coriolis.grad_f_cross_u(xi,ui) - ddot(i) - g*laps(i))*dt;
    }
    dh(i) = (-sigma(i) * h(i))*dt;
  }
};


/** @brief  RHS tendencies for the shallow water equations
  at active particles  (both planar and spherical problems)

  If U is the state vector, and the SWE are written as a dynamical
  system, dU/dt = F(U), this functor computes the RHS F, given U.

  This functor will be called from a Kokkos::parallel_for, with a range
  policy over all active particles.
*/
template <typename Geo>
struct SWEVorticityDivergenceAreaTendencies {
  using crd_view = typename Geo::crd_view_type;
  using vec_view = typename Geo::vec_view_type;
  using coriolis_type = typename std::conditional<
    std::is_same<Geo, PlaneGeometry>::value, CoriolisBetaPlane,
      CoriolisSphere>::type;
  scalar_view_type dzeta; /// [out] vorticity tendency
  scalar_view_type dsigma; /// [out] divergence tendency
  scalar_view_type darea; /// [out] area tendency
  crd_view x; /// [in] physical coordinates
  vec_view u; /// [in] velcocity
  scalar_view_type zeta; /// [in] vorticity
  scalar_view_type sigma; /// [in] divergences
  scalar_view_type area; /// [in] areas
  scalar_view_type ddot; /// [in] double dot product
  scalar_view_type laps; /// [in] surface Laplacian
  coriolis_type coriolis; /// [in] Coriolis functor
  Real g; /// [in] gravity
  Real dt; /// [in] time step size

  SWEVorticityDivergenceAreaTendencies(
    scalar_view_type& dzeta,
    scalar_view_type& dsigma, scalar_view_type& darea,
    const crd_view& x, const vec_view& u,
    const scalar_view_type zeta, const scalar_view_type sigma,
    const scalar_view_type area, const scalar_view_type ddot,
    const scalar_view_type laps,
    const coriolis_type& cor, const Real g, const Real dt) :
    dzeta(dzeta),
    dsigma(dsigma),
    darea(darea),
    x(x),
    u(u),
    zeta(zeta),
    sigma(sigma),
    area(area),
    ddot(ddot),
    laps(laps),
    coriolis(cor),
    g(g),
    dt(dt) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index i) const {
    const auto xi = Kokkos::subview(x, i, Kokkos::ALL);
    const auto ui = Kokkos::subview(u, i, Kokkos::ALL);
    const Real f = coriolis.f(xi);
    dzeta(i) = (-coriolis.dfdt(ui) - (zeta(i) + f) * sigma(i)) * dt;
    if constexpr (std::is_same<Geo,SphereGeometry>::value) {
      dsigma(i) = (f*zeta(i) + coriolis.grad_f_cross_u(xi, ui)
        - ddot(i) - g*laps(i) - SphereGeometry::norm2(ui)) * dt;
    }
    else {
      dsigma(i) = (f*zeta(i) + coriolis.grad_f_cross_u(xi, ui) - ddot(i) - g*laps(i)) * dt;
    }
    darea(i) = (sigma(i) * area(i)) * dt;
  }
};

/** @brief  Given depth, this functor sets the surface height
  for a bottom topography defined by TopoType.

  For SWE (both planar and spherical) problems.

  This functor will be called from a Kokkos::parallel_for, with a range
  policy over all passive particles.
*/
template <typename Geo, typename TopoType>
struct SetSurfaceFromDepth {
  using crd_view = typename Geo::crd_view_type;

  scalar_view_type s; /// [out] surface height at passive particles
  scalar_view_type b; /// [out] bottom height at passive particles
  crd_view x; /// [in] position of passive particles
  scalar_view_type h; /// [in] depth at passive particles
  TopoType topo; /// [in] bottom topography functor

  SetSurfaceFromDepth(scalar_view_type& s, scalar_view_type& b,
    const crd_view x, const scalar_view_type h, const TopoType& topo) :
    s(s), b(b), x(x), h(h) {}


  KOKKOS_INLINE_FUNCTION
  void operator() (const Index i) const {
    const auto xi = Kokkos::subview(x, i, Kokkos::ALL);
    b(i) = topo(xi);
    s(i) = h(i) + b(i);
  }
};


/** @brief  Given mass and area, this functor sets the surface height
  for a bottom topography defined by TopoType.

  For SWE (both planar and spherical) problems.

  This functor will be called from a Kokkos::parallel_for, with a range
  policy over all active particles.
*/
template <typename Geo, typename TopoType>
struct SetDepthAndSurfaceFromMassAndArea {
  using crd_view = typename Geo::crd_view_type;

  scalar_view_type h; /// [out] depth at active particles
  scalar_view_type s; /// [out] surface height at active particles
  scalar_view_type b; /// [out] bottom height at active particles
  crd_view x; /// [in] position of active particles
  scalar_view_type m; /// [in] mass of active particles
  scalar_view_type area; /// [in] area of active particles
  mask_view_type mask; /// [in] mask to exclude divided panels
  TopoType topo; /// [in] bottom topography functor

  SetDepthAndSurfaceFromMassAndArea(scalar_view_type& h,
                    scalar_view_type& s,
                    scalar_view_type& b,
                    const crd_view x,
                    const scalar_view_type m,
                    const scalar_view_type area,
                    const mask_view_type mask,
                    const TopoType topo) :
        h(h), s(s), b(b), x(x), m(m), area(area), mask(mask) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index i) const {
    if (!mask(i)) {
      h(i) = m(i) / area(i);
      const auto xi = Kokkos::subview(x, i, Kokkos::ALL);
      b(i) = topo(xi);
      s(i) = b(i) + h(i);
    }
  }
};



} // namespace Lpm

#endif
