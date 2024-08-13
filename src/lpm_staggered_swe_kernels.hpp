#ifndef LPM_STAGGERED_SWE_KERNELS_HPP
#define LPM_STAGGERED_SWE_KERNELS_HPP

#include "LpmConfig.h"
#include "lpm_assert.hpp"
#include "lpm_constants.hpp"
#include "lpm_coriolis.hpp"
#include "lpm_geometry.hpp"
#include "util/lpm_floating_point.hpp"
#include "util/lpm_math.hpp"
#include "util/lpm_tuple.hpp"

namespace Lpm {

template <typename Geo>
struct StaggeredSWETendencies {
  using crd_view = typename Geo::crd_view_type;
  using vec_view = typename Geo::vec_view_type;
  using coriolis_type = typename std::conditional<
    std::is_same<Geo, PlaneGeometry>::value, CoriolisBetaPlane,
      CoriolisSphere>::type;

  scalar_view_type dzeta; // output
  scalar_view_type dsigma; // output
  scalar_view_type darea; // output
  crd_view x; // input
  vec_view u; // input;
  scalar_view_type zeta; //input
  scalar_view_type sigma; // input
  scalar_view_type grad_f_cross_u; // input
  scalar_view_type ddot; // input
  scalar_view_type laps; // input
  coriolis_type coriolis; //input
  Real g; // input
  Real dt; // input

  StaggeredSWETendencies(scalar_view_type& dz, scalar_view_type& ds,
    scalar_view_type& da, const crd_view x, const vec_view u,
    const scalar_view_type zeta, const scalar_view_type sigma,
    const scalar_view_type area,
    const scalar_view_type gfcu, const scalar_view_type ddot,
    const scalar_view_type laps, const mask_view_type mask,
    const coriolis_type& coriolis,
    const Real g, const Real dt) :
    dzeta(dz),
    dsigma(ds),
    darea(da),
    x(x),
    u(u),
    zeta(zeta),
    sigma(sigma),
    area(area),
    grad_f_cross_u(gfcu),
    ddot(ddot),
    laps(laps),
    mask(mask),
    coriolis(coriolis),
    g(g),
    dt(dt) {}


  KOKKOS_INLINE_FUNCTION
  void operator() (const Index i) const {
    const auto xi = Kokkos::subview(x, i, Kokkos::ALL);
    const auto ui = Kokkos::subview(u, i, Kokkos::ALL);
    const Real f = coriolis.f(xi);
    dzeta(i) = (-coriolis.dfdt(ui) - (zeta + f)*sigma(i)) * dt;
    dsigma(i) = (f*zeta(i) + grad_f_cross_u(i) - ddot(i) - laps(i))*dt;
    darea(i) = (sigma(i) * area(i))*dt;
  }
};

template <typename Geo>
struct StaggeredSWEKernels {
  static constexpr Int vec_size = Geo::ndim;
  static constexpr Int mat_size = Geo::ndim * Geo::ndim;
  static constexpr Int n_packed_vals = 2*vec_size + 2*mat_size;
  using value_type = Kokkos::Tuple<Real, n_packed_vals>;
  using vec_type = Kokkos::Tuple<Real, vec_size>;
  using mat_type = Kokkos::Tuple<Real, mat_size>;

  template <typename XType, typename YType>
  KOKKOS_INLINE_FUNCTION
  static Real greens(const XType& x, const YType& y, const Real& eps) {
    const Real arg = 1 - SphereGeometry::dot(x, y) + square(eps);
    return arg / (-4*constants::PI);
  }

  template <typename XType, typename YType>
  KOKKOS_INLINE_FUNCTION
  static vec_type kzeta(const XType& x, const YType& y, const Real& eps) {
    const Real denom = -4*constants::PI *(1 - SphereGeometry::dot(x, y) + square(eps));
    vec_type result;
    SphereGeometry::cross(result, x, y);
    for (int k=0; k<3; ++k) {
      result[k] /= denom;
    }
    return result;
  }

  template <typename XType, typename YType>
  KOKKOS_INLINE_FUNCTION
  static vec_type ksigma(const XType& x, const YType& y, const Real& eps) {
    const Real denom = -4*constants::PI *(1 - SphereGeometry::dot(x, y) + square(eps));
    const mat_type proj_mat = spherical_tangent_projection_matrix(x);
    vec_type result;
    apply_3by3(result, proj_mat, y);
    for (int k=0; k<3; ++k) {
      result[k] /= denom;
    }
    return result;
  }

  template <typename XType, typename YType>
  KOKKOS_INLINE_FUNCTION
  static mat_type grad_kzeta(const XType& x, const YType& y, const Real& eps) {
    const mat_type proj_mat = spherical_tangent_projection_matrix(x);
    const Real denom = 4*constants::PI *(1 - SphereGeometry::dot(x, y) + square(eps));
    const mat_type permuted_y = {    0,  y[2], -y[1],
                                 -y[2],     0,  y[0],
                                  y[1], -y[0],    0};
    mat_type term1;
    matmul_3by3(term1, proj_mat, permuted_y);
    for (int k=0; k<mat_size; ++k) {
      term1[k] /= denom;
    }
    const vec_type ks = ksigma(x, y, eps);
    const vec_type kz =  kzeta(x, y, eps);
    mat_type term2;
    outer_product_r3(term2, ks, kz);
    for (int k=0; k<mat_size; ++k) {
      term2 *= 4*constants::PI;
    }
    return term1 + term2;
  }

  template <typename XType, typename YType>
  KOKKOS_INLINE_FUNCTION
  static mat_type grad_ksigma(const XType& x, const YType& y, const Real& eps) {
    const mat_type proj_mat = spherical_tangent_projection_matrix(x);
    const Real denom = 4*constants::PI *(1 - SphereGeometry::dot(x, y) + square(eps));
  }
};



} // namespace Lpm

#endif
