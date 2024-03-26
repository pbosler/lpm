#ifndef LPM_SWE_KERNELS_HPP
#define LPM_SWE_KERNELS_HPP

#include "LpmConfig.h"
#include "lpm_geometry.hpp"
#include "lpm_pse.hpp"
#include "util/lpm_math.hpp"

namespace Lpm {

/** @brief Evaluates the Biot-Savart kernel in the plane.
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

template <typename Compressed3by3>
KOKKOS_INLINE_FUNCTION
Real double_dot(const Compressed3by3& mat) {
  Real result = 0;
  for (Int i=0; i<3; ++i) {
    for (Int j=i; j<3; ++i) {
      const Int ij_idx = 3*i + j;
      const Int ji_idx = 3*j + i;
      result += (i==j ? 1 : 2) * mat[ij_idx] * mat[ji_idx];
    }
  }
  return result;
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

/** Evaluates the RHS terms for the planar shallow water equations
  using Particle Strength Exchange (PSE) to compute the Laplacian
  of the fluid surface.

  Each call to this function computes 1 pairwise interaction, the
  contributions of source particle y to target particle x.

  Results are packed into a 7-tuple:
  index 0: u
  index 1: v
  index 2: du/dx
  index 3: du/dy
  index 4: dv/dx
  index 5: dv/dy
  index 6: laplacian(sfc) from PSE
*/
template <typename XType, typename YType>
KOKKOS_INLINE_FUNCTION
Kokkos::Tuple<Real, 7> planar_swe_sums_rhs_pse(
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
  using pse_type = pse::BivariateOrder8<PlaneGeometry>;
  Kokkos::Tuple<Real, 7> result;

  const Real Rvec[2] = {-(tgt_x[1] - src_y[1]), tgt_x[0] - src_y[0]};
  const Real Svec[2] = {  tgt_x[0] - src_y[0] , tgt_x[1] - src_y[1]};
  const Real denom = PlaneGeometry::norm2(Svec) + square(eps);
  const Real rot_str = src_zeta * src_area;
  const Real pot_str = src_sigma * src_area;
  constexpr Real oo2pi = 1.0 / (2*constants::PI);
  // u
  result[0] = oo2pi * (Rvec[0]*rot_str + Svec[0]*pot_str) / denom;
  // v
  result[1] = oo2pi * (Rvec[1]*rot_str + Svec[1]*pot_str) / denom;
  // du/dx
  result[2] = oo2pi * ( pot_str / denom - (2*Svec[0]/square(denom)) * (Rvec[0]*rot_str + Svec[0]*pot_str));
  // du/dy
  result[3] = oo2pi * (-rot_str / denom - (2*Svec[1]/square(denom)) * (Rvec[0]*rot_str + Svec[0]*pot_str));
  // dv/dx
  result[4] = oo2pi * ( rot_str / denom - (2*Svec[0]/square(denom)) * (Rvec[1]*rot_str + Svec[1]*pot_str));
  // dv/dy
  result[5] = oo2pi * ( pot_str / denom - (2*Svec[1]/square(denom)) * (Rvec[1]*rot_str + Svec[1]*pot_str));
  // lap(s)
  const Real pse_input = pse_type::kernel_input(tgt_x, src_y, pse_eps);
  const Real pse_val = pse_type::laplacian(pse_input);
  result[6] = (src_s - tgt_s) * src_area * pse_val / square(pse_eps);

  return result;
}

/** @brief Customized Kokkos Reducer for planar SWE direct summation,
  with Particle Strength Exchange for the Laplacian.

  This functor will be called from a Kokkos::parallel_reduce kernel.
*/
struct PlanarSwePseDirectSumReducer {
  using crd_view = PlaneGeometry::crd_view_type;
  using value_type = Kokkos::Tuple<Real,7>;
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
    if (!collocated_src_tgt or i != j) {
      if (!src_mask(j)) {
        const auto xcrd = Kokkos::subview(tgt_x, i, Kokkos::ALL);
        const auto ycrd = Kokkos::subview(src_y, j, Kokkos::ALL);
        const value_type local_result =
          planar_swe_sums_rhs_pse(xcrd, ycrd, src_zeta(j), src_sigma(j),
            src_area(j), src_sfc(j), tgt_sfc(i), eps, pse_eps);

        r += local_result;
      }
    }
  }
};

/** @brief Direct sum functor for Planar SWE vertices (passive particles).
*/
struct PlanarSWEVertexSums {
  using vec_view = PlaneGeometry::vec_view_type;
  using crd_view = PlaneGeometry::crd_view_type;

  vec_view vert_u; /// [out] velocity at vertices
  scalar_view_type vert_ddot; /// [out] double dot product at vertices
  scalar_view_type vert_laps; /// [out] surface Laplacian at vertices
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

  PlanarSWEVertexSums(vec_view& vu, scalar_view_type& vdd, scalar_view_type& vls,
    const crd_view vx, const scalar_view_type vsfc, const crd_view fy,
    const scalar_view_type fzeta, const scalar_view_type fsig,
    const scalar_view_type farea, const mask_view_type fmask,
    const scalar_view_type fsfc, const Real eps, const Real pse_eps,
    const Index nfaces) :
    vert_u(vu),
    vert_ddot(vdd),
    vert_laps(vls),
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
    nfaces(nfaces) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const member_type& thread_team) const {
    const Index i = thread_team.league_rank();

    // perform packed reduction, dispatching 1 thread team per target
    Kokkos::Tuple<Real,7> sums;
    constexpr bool collocated = false;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(thread_team, nfaces),
      PlanarSwePseDirectSumReducer(vert_x, vert_s, i, face_y, collocated,
        face_zeta, face_sigma, face_area, face_mask, face_s,
        eps, pse_eps), sums);

    // unpack reduction results
    vert_u(i,0) = sums[0];
    vert_u(i,1) = sums[1];
    vert_ddot(i) = square(sums[2]) + 2*sums[3]*sums[4] + square(sums[5]);
    vert_laps(i) = sums[6];
  }
};

/** @brief Direct sum functor for Planar SWE (active particles).
*/
struct PlanarSWEFaceSums {
  using vec_view = PlaneGeometry::vec_view_type;
  using crd_view = PlaneGeometry::crd_view_type;

  vec_view face_u; /// [out] velocity at faces (targets)
  scalar_view_type face_ddot; /// [out] double dot product at faces
  scalar_view_type face_laps; /// [out] surface Laplacian at faces
  crd_view face_xy; /// [in] coordinates of faces (sources and targets)
  scalar_view_type face_zeta; /// [in] vorticity at faces
  scalar_view_type face_sigma; /// [in] divergence at faces
  scalar_view_type face_area; /// [in] face area
  mask_view_type face_mask; /// [in] mask to exclude divided panels
  scalar_view_type face_s; /// [in] surface height at faces
  Real eps; /// [in] velocity kernel smoothing parameter
  Real pse_eps; /// [in] PSE kernel width parameter
  Index nfaces; /// [in] total number of faces (including divided faces)

  PlanarSWEFaceSums(vec_view& fu, scalar_view_type& fdd, scalar_view_type& fls,
    const crd_view fxy, const scalar_view_type fzeta, const scalar_view_type fsig,
    const scalar_view_type farea, const mask_view_type fmask, const scalar_view_type fsfc,
    const Real eps, const Real pse_eps, const Index nfaces) :
    face_u(fu),
    face_ddot(fdd),
    face_laps(fls),
    face_xy(fxy),
    face_zeta(fzeta),
    face_sigma(fsig),
    face_area(farea),
    face_mask(fmask),
    face_s(fsfc),
    eps(eps),
    pse_eps(pse_eps),
    nfaces(nfaces) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const member_type& thread_team) const {
    const Index i = thread_team.league_rank();

    // perform packed reduction, 1 thread team per target
    Kokkos::Tuple<Real,7> sums;
    const bool collocated = !(eps > 0);
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(thread_team, nfaces),
      PlanarSwePseDirectSumReducer(face_xy, face_s, i, face_xy, collocated,
        face_zeta, face_sigma, face_area, face_mask, face_s,
        eps, pse_eps), sums);

    // unpack reduction results
    face_u(i,0) = sums[0];
    face_u(i,1) = sums[1];
    face_ddot(i) = square(sums[2]) + 2*sums[3]*sums[4] + square(sums[5]);
    face_laps(i) = sums[6];
  }
};

struct PlanarSWEVorticityDivergenceHeightTendencies {
  scalar_view_type dzeta; /// [out] vorticity tendency
  scalar_view_type dsigma; /// [out] divergence tendency
  scalar_view_type dh; /// [out] depth tendency
  scalar_view_type zeta; /// [in] vorticity
  scalar_view_type sigma; /// [in] divergence
  scalar_view_type h; /// [in] depth
  scalar_view_type ddot; /// [in] double dot product
  scalar_view_type laps; /// [in] surface Laplacian
  Real f; /// [in] Coriolis (f-plane) parameter
  Real g; /// [in] gravity
  Real dt; /// [in] time step size

  PlanarSWEVorticityDivergenceHeightTendencies(scalar_view_type& dzeta,
    scalar_view_type& dsigma, scalar_view_type& dh,
    const scalar_view_type zeta, const scalar_view_type sigma,
    const scalar_view_type h, const scalar_view_type ddot,
    const scalar_view_type laps, const Real f, const Real g, const Real dt) :
    dzeta(dzeta),
    dsigma(dsigma),
    dh(dh),
    zeta(zeta),
    sigma(sigma),
    h(h),
    ddot(ddot),
    laps(laps),
    f(f),
    g(g),
    dt(dt) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index i) const {
    dzeta(i) = ((zeta(i) + f) * sigma(i))*dt;
    dsigma(i) = (f*zeta(i) - ddot(i) - g*laps(i))*dt;
    dh(i) = (-sigma(i) * h(i))*dt;
  }
};

struct PlanarSWEVorticityDivergenceAreaTendencies {
  scalar_view_type dzeta; /// [out] vorticity tendency
  scalar_view_type dsigma; /// [out] divergence tendency
  scalar_view_type darea; /// [out] area tendency
  scalar_view_type zeta; /// [in] vorticity
  scalar_view_type sigma; /// [in] divergences
  scalar_view_type area; /// [in] areas
  scalar_view_type ddot; /// [in] double dot product
  scalar_view_type laps; /// [in] surface Laplacian
  Real f; /// [in] Coriolis (f-plane) parameter
  Real g; /// [in] gravity
  Real dt; /// [in] time step size

  PlanarSWEVorticityDivergenceAreaTendencies(
    scalar_view_type& dzeta,
    scalar_view_type& dsigma, scalar_view_type& darea,
    const scalar_view_type zeta, const scalar_view_type sigma,
    const scalar_view_type area, const scalar_view_type ddot,
    const scalar_view_type laps, const Real f, const Real g, const Real dt) :
    dzeta(dzeta),
    dsigma(dsigma),
    darea(darea),
    zeta(zeta),
    sigma(sigma),
    area(area),
    ddot(ddot),
    laps(laps),
    f(f),
    g(g),
    dt(dt) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index i) const {
    dzeta(i) = ((zeta(i) + f) * sigma(i)) * dt;
    dsigma(i) = (f*zeta(i) - ddot(i) - g*laps(i)) * dt;
    darea(i) = (sigma(i) * area(i)) * dt;
  }
};

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
    const auto mx = Kokkos::subview(x, i, Kokkos::ALL);
    b(i) = topo(mx);
    s(i) = h(i) + b(i);
  }
};

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
      const auto mx = Kokkos::subview(x, i, Kokkos::ALL);
      b(i) = topo(mx);
      s(i) = b(i) + h(i);
    }
  }
};



} // namespace Lpm

#endif
