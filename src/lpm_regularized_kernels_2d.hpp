#ifndef LPM_REGULARIZED_KERNELS_2D_HPP
#define LPM_REGULARIZED_KERNELS_2D_HPP

#include "LpmConfig.h"
#include "lpm_constants.hpp"
#include "lpm_geometry.hpp"
#include "util/lpm_math.hpp"
#include "util/lpm_tuple.hpp"

namespace Lpm {
namespace colloc {

namespace impl {
template <typename XType, typename KernelType> KOKKOS_INLINE_FUNCTION
Kokkos::Tuple<Real,6> plane_velocity_kernel_impl(const XType& x, const KernelType& kernels) {
  const Real xscaled[2] = {x[0]/kernels.eps, x[1]/kernels.eps};
  const Real one_over_rsq = FloatingPoint<Real>::safe_denominator(PlaneGeometry::norm2(x));
  const Real q = kernels.blobq(xscaled);
  const Real ucoeff = q * one_over_rsq / (2*constants::PI);
  const Real grad_coeff = -(kernels.scaled_blob(x) - q * one_over_rsq /constants::PI) * one_over_rsq;
  const Real grad_diag = q * one_over_rsq / (2*constants::PI);
  Kokkos::Tuple<Real,6> u;
  u[0] = -x[0] * ucoeff;
  u[1] = -x[1] * ucoeff;
  u[2] = grad_coeff * square(x[0]) - grad_diag;
  u[3] = grad_coeff * x[0] * x[1];
  u[4] = grad_coeff * x[1] * x[0];
  u[5] = grad_coeff * square(x[1]) - grad_diag;
  return u;
}

template <typename XType, typename KernelType> KOKKOS_INLINE_FUNCTION
Real plane_scaled_blob_impl(const XType& x, const KernelType& kernels) {
  const Real xscaled[2] = {x[0]/kernels.eps, x[1]/kernels.eps};
  return kernels.blob(xscaled)/kernels.epssq;
}

} // namespace impl

/**  Each kernel set provides "blob" functions for interpolation,
  velocity kernels, and derivatives such as the laplacian, all
  with the same order.
*/



/** 2nd order kernels for the plane
*/
struct Plane2ndOrder {
  Real eps;
  Real epssq;
  static constexpr Real denom = constants::PI;
  Real scaled_denom;
  static constexpr Int order = 2;

  KOKKOS_INLINE_FUNCTION
  explicit Plane2ndOrder(const Real epsilon) :
    eps(epsilon),
    epssq(square(epsilon)),
    scaled_denom(constants::PI * square(epsilon)) {};

  KOKKOS_INLINE_FUNCTION
  Plane2ndOrder(const Plane2ndOrder& other) = default;

  template <typename XType> KOKKOS_INLINE_FUNCTION
   Real blob(const XType& x) const {
    const Real rsq = PlaneGeometry::norm2(x);
    return exp(-rsq) / denom;
  }

  template <typename XType> KOKKOS_INLINE_FUNCTION
   Real scaled_blob(const XType& x) const {
    return impl::plane_scaled_blob_impl(x, *this);
  }

  template <typename XType> KOKKOS_INLINE_FUNCTION
  Real blobq(const XType& x) const {
    const Real rsq = PlaneGeometry::norm2(x);
    return 1 - exp(-rsq);
  }

  /** Returns a packed reduction with velocity and the velocity gradient tensor.

    u1 = v[0]
    u2 = v[1];
    du1/dx1 = v[2];
    du1/dx2 = v[3];
    du2/dx1 = v[4];
    du2/dx2 = v[5];
  */
  template <typename XType> KOKKOS_INLINE_FUNCTION
  Kokkos::Tuple<Real,6> velocity(const XType& x) const {
    return impl::plane_velocity_kernel_impl(x, *this);
  }

  template <typename XType> KOKKOS_INLINE_FUNCTION
   Real x0_derivative(const XType& x) const {
    const Real xscaled[2] = {x[0]/eps, x[1]/eps};
    const Real rsq = PlaneGeometry::norm2(xscaled);
    const Real coeff = -2 / scaled_denom;
    return xscaled[0] * coeff * exp(-rsq);
  }

  template <typename XType> KOKKOS_INLINE_FUNCTION
   Real x1_derivative(const XType& x) const {
    const Real xscaled[2] = {x[0]/eps, x[1]/eps};
    const Real rsq = PlaneGeometry::norm2(xscaled);
    const Real coeff =  -2 / scaled_denom;
    return xscaled[1] * coeff * exp(-rsq);
  }

  template <typename XType> KOKKOS_INLINE_FUNCTION
   Real left_x0_derivative(const XType& x) const {
    const Real xscaled[2] = {x[0]/eps, x[1]/eps};
    const Real rsq = PlaneGeometry::norm2(xscaled);
    const Real coeff = (-20 + 8*rsq) / scaled_denom;
    return xscaled[0] * coeff * exp(-rsq);
  }

  template <typename XType> KOKKOS_INLINE_FUNCTION
   Real left_x1_derivative(const XType& x) const {
    const Real xscaled[2] = {x[0]/eps, x[1]/eps};
    const Real rsq = PlaneGeometry::norm2(xscaled);
    const Real coeff = (-20 + 8*rsq) / scaled_denom;
    return xscaled[1] * coeff * exp(-rsq);
  }

  template <typename XType> KOKKOS_INLINE_FUNCTION
   Real laplacian(const XType& x) const {
    const Real xscaled[2] = {x[0]/eps, x[1]/eps};
    const Real rsq = PlaneGeometry::norm2(xscaled);
    const Real coeff = 4 / scaled_denom;
    return coeff * exp(-rsq);
  }
};

struct Plane4thOrder {
  Real eps;
  Real epssq;
  static constexpr Real denom = constants::PI;
  Real scaled_denom;
  static constexpr Int order = 4;

  KOKKOS_INLINE_FUNCTION
  explicit Plane4thOrder(const Real epsilon) :
    eps(epsilon),
    epssq(square(epsilon)),
    scaled_denom(constants::PI * square(epsilon)) {};

  KOKKOS_INLINE_FUNCTION
  Plane4thOrder(const Plane4thOrder& other) = default;

  template <typename XType> KOKKOS_INLINE_FUNCTION
  Real blob(const XType& x) const {
      const Real rsq = PlaneGeometry::norm2(x);
      const Real coeff = ( 2 - rsq ) / denom;
    return coeff * exp(-rsq);
  }

  template <typename XType> KOKKOS_INLINE_FUNCTION
  Real blobq(const XType& x) const {
    const Real rsq = PlaneGeometry::norm2(x);
    return 1 - (1-rsq)*exp(-rsq);
  }

  template <typename XType> KOKKOS_INLINE_FUNCTION
  Real scaled_blob(const XType& x) const {
    return impl::plane_scaled_blob_impl(x, *this);
  }

  /** Returns a packed reduction with velocity and the velocity gradient tensor.

    u1 = v[0]
    u2 = v[1];
    du1/dx1 = v[2];
    du1/dx2 = v[3];
    du2/dx1 = v[4];
    du2/dx2 = v[5];
  */
  template <typename XType> KOKKOS_INLINE_FUNCTION
  Kokkos::Tuple<Real,6> velocity(const XType& x) const {
    return impl::plane_velocity_kernel_impl(x, *this);
  }

  template <typename XType> KOKKOS_INLINE_FUNCTION
  Real laplacian(const XType& x) const {
    const Real xscaled[2] = {x[0]/eps, x[1]/eps};
    const Real rsq = PlaneGeometry::norm2(xscaled);
    const Real coeff = (12 - 4 * rsq ) / scaled_denom;
    return coeff * exp(-rsq);
  }

  template <typename XType> KOKKOS_INLINE_FUNCTION
   Real x0_derivative(const XType& x) const  {
    const Real xscaled[2] = {x[0]/eps, x[1]/eps};
    const Real rsq = PlaneGeometry::norm2(xscaled);
    const Real coeff = xscaled[0] * ( -6 + 2*rsq ) / scaled_denom;
    return coeff * exp(-rsq);
  }

  template <typename XType> KOKKOS_INLINE_FUNCTION
   Real x1_derivative(const XType& x) const {
    const Real xscaled[2] = {x[0]/eps, x[1]/eps};
    const Real rsq = PlaneGeometry::norm2(xscaled);
    const Real coeff = xscaled[1] * ( -6 + 2*rsq ) / scaled_denom;
    return coeff * exp(-rsq);
  }

  template <typename XType> KOKKOS_INLINE_FUNCTION
   Real left_x0_derivative(const XType& x) const  {
    const Real xscaled[2] = {x[0]/eps, x[1]/eps};
    const Real rsq = PlaneGeometry::norm2(xscaled);
    const Real r4th = square(rsq);
    const Real coeff = (-140 + 196*rsq - 64*r4th + 16*rsq*r4th/3) / scaled_denom;
    return xscaled[0] * coeff * exp(-rsq);
  }

  template <typename XType> KOKKOS_INLINE_FUNCTION
   Real left_x1_derivative(const XType& x) const {
    const Real xscaled[2] = {x[0]/eps, x[1]/eps};
    const Real rsq = PlaneGeometry::norm2(xscaled);
    const Real r4th = square(rsq);
    const Real coeff = (-140 + 196*rsq - 64*r4th + 16*rsq*r4th/3) / scaled_denom;
    return xscaled[1] * coeff * exp(-rsq);
  }
};

struct Plane6thOrder {
  Real eps;
  Real epssq;
  static constexpr Real denom = constants::PI;
  Real scaled_denom;
  static constexpr Int order = 6;

  KOKKOS_INLINE_FUNCTION
  explicit Plane6thOrder(const Real epsilon) :
    eps(epsilon),
    epssq(square(epsilon)),
    scaled_denom(constants::PI * square(epsilon)) {};

  KOKKOS_INLINE_FUNCTION
  Plane6thOrder(const Plane6thOrder& other) = default;

  template <typename XType> KOKKOS_INLINE_FUNCTION
   Real blob(const XType& x) const {
    const Real rsq = PlaneGeometry::norm2(x);
    const Real r4th = square(rsq);
    const Real coeff = ( 3 - 3*rsq + r4th/2 ) / denom;
    return coeff * exp(-rsq);
  }

  template <typename XType> KOKKOS_INLINE_FUNCTION
   Real scaled_blob(const XType& x) const {
    return impl::plane_scaled_blob_impl(x, *this);
  }

  template <typename XType> KOKKOS_INLINE_FUNCTION
  Real blobq(const XType& x) const {
    const Real rsq = PlaneGeometry::norm2(x);
    return 1 - (1 - 2*rsq + 0.5*square(rsq))*exp(-rsq);
  }

  template <typename XType> KOKKOS_INLINE_FUNCTION
   Real laplacian(const XType& x) const {
    const Real xscaled[2] = {x[0]/eps, x[1]/eps};
    const Real rsq = PlaneGeometry::norm2(xscaled);
    const Real r4th = square(rsq);
    const Real coeff = (24 - 16*rsq + 2*r4th ) / scaled_denom;
    return coeff * exp(-rsq);
  }

  template <typename XType> KOKKOS_INLINE_FUNCTION
   Real x0_derivative(const XType& x) const {
    const Real xscaled[2] = {x[0]/eps, x[1]/eps};
    const Real rsq = PlaneGeometry::norm2(xscaled);
    const Real r4th = square(rsq);
    const Real coeff = xscaled[0] * ( -12 + 8*rsq - r4th ) / scaled_denom;
    return coeff * exp(-rsq);
  }

   template <typename XType> KOKKOS_INLINE_FUNCTION
   Real x1_derivative(const XType& x) const {
    const Real xscaled[2] = {x[0]/eps, x[1]/eps};
    const Real rsq = PlaneGeometry::norm2(xscaled);
    const Real r4th = square(rsq);
    const Real coeff = xscaled[1] * ( -12 + 8*rsq - r4th ) / scaled_denom;
    return coeff * exp(-rsq);
  }

  /** Returns a packed reduction with velocity and the velocity gradient tensor.

    u1 = v[0]
    u2 = v[1];
    du1/dx1 = v[2];
    du1/dx2 = v[3];
    du2/dx1 = v[4];
    du2/dx2 = v[5];
  */
  template <typename XType> KOKKOS_INLINE_FUNCTION
  Kokkos::Tuple<Real,6> velocity(const XType& x) const {
    return impl::plane_velocity_kernel_impl(x, *this);
  }

  template <typename XType> KOKKOS_INLINE_FUNCTION
   Real left_x0_derivative(const XType& x) const  {
    const Real xscaled[2] = {x[0]/eps, x[1]/eps};
    const Real rsq = PlaneGeometry::norm2(xscaled);
    const Real r4th = square(rsq);
    const Real coeff = (-504 + 1344*rsq - 978*r4th+268*rsq*r4th-88*square(r4th)/3 + 16*square(r4th)*rsq/15) / scaled_denom;
    return xscaled[0] * coeff * exp(-rsq);
  }

  template <typename XType> KOKKOS_INLINE_FUNCTION
   Real left_x1_derivative(const XType& x) const  {
    const Real xscaled[2] = {x[0]/eps, x[1]/eps};
    const Real rsq = PlaneGeometry::norm2(xscaled);
    const Real r4th = square(rsq);
    const Real coeff = (-504 + 1344*rsq - 978*r4th+268*rsq*r4th-88*square(r4th)/3 + 16*square(r4th)*rsq/15) / scaled_denom;
    return xscaled[1] * coeff * exp(-rsq);
  }};

struct Plane8thOrder {
  Real eps;
  Real epssq;
  Real denom;
  Real scaled_denom;
  static constexpr Int order = 8;

  KOKKOS_INLINE_FUNCTION
  explicit Plane8thOrder(const Real epsilon) :
    eps(epsilon),
    epssq(square(epsilon)),
    denom(constants::PI),
    scaled_denom(constants::PI * square(epsilon)) {};

  KOKKOS_INLINE_FUNCTION
  Plane8thOrder(const Plane8thOrder& other) = default;

  template <typename XType> KOKKOS_INLINE_FUNCTION
   Real blob(const XType& x) const {
    const Real rsq = PlaneGeometry::norm2(x);
    const Real r4th = square(rsq);
    const Real coeff = ( 4 - 6*rsq + 2*r4th - rsq*r4th / 6 ) / denom;
    return coeff * exp(-rsq);
  }

  template <typename XType> KOKKOS_INLINE_FUNCTION
   Real scaled_blob(const XType& x) const {
    return impl::plane_scaled_blob_impl(x, *this);
  }

  template <typename XType> KOKKOS_INLINE_FUNCTION
  Real blobq(const XType& x) const {
    const Real rsq = PlaneGeometry::norm2(x);
    const Real r4th = square(rsq);
    const Real coeff = -1 + 3*rsq - 1.5*r4th + r4th*rsq/6;
    return 1 + coeff*exp(-rsq);
  }

  template <typename XType> KOKKOS_INLINE_FUNCTION
   Real laplacian(const XType& x) const {
    const Real xscaled[2] = {x[0]/eps, x[1]/eps};
    const Real rsq = PlaneGeometry::norm2(xscaled);
    const Real r4th = square(rsq);
    const Real coeff = (40 - 40 * rsq + 10*r4th - 2*rsq*r4th/3  ) / scaled_denom;
    return coeff * exp(-rsq);
  }

  template <typename XType> KOKKOS_INLINE_FUNCTION
   Real x0_derivative(const XType& x) const {
    const Real xscaled[2] = {x[0]/eps, x[1]/eps};
    const Real rsq = PlaneGeometry::norm2(xscaled);
    const Real r4th = square(rsq);
    const Real coeff = xscaled[0] * ( -20 + 20*rsq - 5 * r4th + rsq*r4th/3 ) / scaled_denom;
    return coeff * exp(-rsq);
  }

  template <typename XType> KOKKOS_INLINE_FUNCTION
   Real x1_derivative(const XType& x) const {
    const Real xscaled[2] = {x[0]/eps, x[1]/eps};
    const Real rsq = PlaneGeometry::norm2(xscaled);
    const Real r4th = square(rsq);
    const Real coeff = xscaled[1] * ( -20 + 20*rsq - 5 * r4th + rsq*r4th/3 ) / scaled_denom;
    return coeff * exp(-rsq);
  }

  /** Returns a packed reduction with velocity and the velocity gradient tensor.

    u1 = v[0]
    u2 = v[1];
    du1/dx1 = v[2];
    du1/dx2 = v[3];
    du2/dx1 = v[4];
    du2/dx2 = v[5];
  */
  template <typename XType> KOKKOS_INLINE_FUNCTION
  Kokkos::Tuple<Real,6> velocity(const XType& x) const {
    return impl::plane_velocity_kernel_impl(x, *this);
  }

  template <typename XType> KOKKOS_INLINE_FUNCTION
  Real left_x0_derivative(const XType& x) const  {
    const Real xscaled[2] = {x[0]/eps, x[1]/eps};
    const Real rsq = PlaneGeometry::norm2(xscaled);
    const Real r4th = square(rsq);
    const Real r6th = rsq*r4th;
    const Real r8th = square(r4th);
    const Real coeff = (-1320 + 5544*rsq -6666*r4th + 9922*r6th/3 - 2336*r8th/3 + 272*r8th*rsq/3 - 224*square(r6th)/45 + 32*r4th*r8th/315) / scaled_denom;
    return xscaled[0] * coeff * exp(-rsq);
  }

  template <typename XType> KOKKOS_INLINE_FUNCTION
  Real left_x1_derivative(const XType& x) const  {
    const Real xscaled[2] = {x[0]/eps, x[1]/eps};
    const Real rsq = PlaneGeometry::norm2(xscaled);
    const Real r4th = square(rsq);
    const Real r6th = rsq*r4th;
    const Real r8th = square(r4th);
    const Real coeff = (-1320 + 5544*rsq -6666*r4th + 9922*r6th/3 - 2336*r8th/3 + 272*r8th*rsq/3 - 224*square(r6th)/45 + 32*r4th*r8th/315) / scaled_denom;
    return xscaled[1] * coeff * exp(-rsq);
  }
};


template <typename KernelType>
struct PlaneVelocityDirectSumReducer {
  using crd_view = PlaneGeometry::crd_view_type;
  using vec_view = PlaneGeometry::vec_view_type;
  using value_type = Kokkos::Tuple<Real,6>;
  using kernel_type = KernelType;
  using view_type = vec_view;
  static constexpr Int ndim = 2;
  crd_view tgt_x;
  crd_view src_x;
  KernelType kernels;
  scalar_view_type vorticity;
  scalar_view_type divergence;
  scalar_view_type area;
  mask_view_type src_mask;
  Index i;

  KOKKOS_INLINE_FUNCTION
  PlaneVelocityDirectSumReducer(const crd_view tgt_x, const crd_view src_x,
    const KernelType& kernels, const scalar_view_type zeta,
    const scalar_view_type delta, const scalar_view_type area,
    const mask_view_type src_mask, const Index i) :
    tgt_x(tgt_x),
    src_x(src_x),
    kernels(kernels),
    vorticity(zeta),
    divergence(delta),
    area(area),
    src_mask(src_mask),
    i(i) {}

  /**
    K1 = v[0]
    K2 = v[1];
    dK1/dx1 = v[2];
    dK1/dx2 = v[3];
    dK2/dx1 = v[4];
    dK2/dx2 = v[5];
  */
  KOKKOS_INLINE_FUNCTION
  void operator() (const Index& j, value_type& u) const {
    if (!src_mask(j)) {
      const auto x_tgt = Kokkos::subview(tgt_x, i, Kokkos::ALL);
      const auto x_src = Kokkos::subview(src_x, j, Kokkos::ALL);
      const Real xij[2] = {x_tgt[0] - x_src[0], x_tgt[1] - x_src[1]};
      const Kokkos::Tuple<Real,6> Kij = kernels.velocity(xij);
      u[0] += ( vorticity(j) * Kij[1] - divergence(j) * Kij[0]) * area(j);
      u[1] += (-vorticity(j) * Kij[0] - divergence(j) * Kij[1]) * area(j);
      u[2] += ( vorticity(j) * Kij[4] - divergence(j) * Kij[2]) * area(j);
      u[3] += ( vorticity(j) * Kij[5] - divergence(j) * Kij[3]) * area(j);
      u[4] += (-vorticity(j) * Kij[2] - divergence(j) * Kij[4]) * area(j);
      u[5] += (-vorticity(j) * Kij[3] - divergence(j) * Kij[5]) * area(j);
    }
  }
};

template <typename KernelType>
struct PlaneScalarInterpolationReducer {
  using crd_view = PlaneGeometry::crd_view_type;
  using vec_view = PlaneGeometry::vec_view_type;
  using value_type = Real;
  using kernel_type = KernelType;
  using view_type = scalar_view_type;
  static constexpr Int ndim = 1;
  crd_view tgt_x;
  scalar_view_type tgt_values;
  crd_view src_x;
  KernelType kernels;
  scalar_view_type src_values;
  scalar_view_type area;
  mask_view_type src_mask;
  Index i;

  KOKKOS_INLINE_FUNCTION
  PlaneScalarInterpolationReducer(const crd_view tgt_x, const scalar_view_type tgt_values,
    const crd_view src_x, const scalar_view_type src_values,
    const KernelType& kernels, const scalar_view_type area,
    const mask_view_type src_mask, const Index i) :
    tgt_x(tgt_x),
    tgt_values(tgt_values),
    src_x(src_x),
    src_values(src_values),
    kernels(kernels),
    area(area),
    src_mask(src_mask),
    i(i) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index& j, Real& f) const {
    if (!src_mask(j)) {
      const auto x_tgt = Kokkos::subview(tgt_x, i, Kokkos::ALL);
      const auto x_src = Kokkos::subview(src_x, j, Kokkos::ALL);
      const Real xij[2] = {x_tgt[0] - x_src[0], x_tgt[1] - x_src[1]};
      f += src_values(j) * area(j) * kernels.scaled_blob(xij);
    }
  }
};

template <typename KernelType>
struct PlaneGradientReducer {
  using crd_view = PlaneGeometry::crd_view_type;
  using vec_view = PlaneGeometry::vec_view_type;
  using value_type = Kokkos::Tuple<Real,2>;
  using kernel_type = KernelType;
  using view_type = vec_view;
  static constexpr Int ndim = 2;
  crd_view tgt_x;
  scalar_view_type tgt_values;
  crd_view src_x;
  scalar_view_type src_values;
  KernelType kernels;
  scalar_view_type area;
  mask_view_type src_mask;
  Index i;

  KOKKOS_INLINE_FUNCTION
  PlaneGradientReducer(const crd_view tgt_x, const scalar_view_type tgt_values,
    const crd_view src_x, const scalar_view_type src_values,
    const KernelType& kernels, const scalar_view_type area,
    const mask_view_type src_mask, const Index i) :
    tgt_x(tgt_x),
    tgt_values(tgt_values),
    src_x(src_x),
    src_values(src_values),
    kernels(kernels),
    area(area),
    src_mask(src_mask),
    i(i) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index& j, value_type& grad) const {
    if (!src_mask(j)) {
      const auto x_tgt = Kokkos::subview(tgt_x, i, Kokkos::ALL);
      const auto x_src = Kokkos::subview(src_x, j, Kokkos::ALL);
      const Real xij[2] = {x_tgt[0] - x_src[0], x_tgt[1] - x_src[1]};
      const Real vij = area(j) * (tgt_values(j) + src_values(i));
      grad[0] += vij * kernels.x0_derivative(xij) / kernels.eps;
      grad[1] += vij * kernels.x1_derivative(xij) / kernels.eps;
    }
  }
};

template <typename KernelType>
struct PlaneLaplacianReducer {
  using crd_view = PlaneGeometry::crd_view_type;
  using vec_view = PlaneGeometry::vec_view_type;
  using value_type = Real;
  using kernel_type = KernelType;
  using view_type = scalar_view_type;
  static constexpr Int ndim = 1;
  crd_view tgt_x;
  scalar_view_type tgt_values;
  crd_view src_x;
  scalar_view_type src_values;
  KernelType kernels;
  scalar_view_type area;
  mask_view_type src_mask;
  Index i;

  KOKKOS_INLINE_FUNCTION
  PlaneLaplacianReducer(const crd_view tgt_x, const scalar_view_type tgt_values,
    const crd_view src_x, const scalar_view_type src_values,
    const KernelType& kernels, const scalar_view_type area,
    const mask_view_type src_mask, const Index i) :
    tgt_x(tgt_x),
    tgt_values(tgt_values),
    src_x(src_x),
    src_values(src_values),
    kernels(kernels),
    area(area),
    src_mask(src_mask),
    i(i) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index& j, Real& lap) const {
    if (!src_mask(j)) {
      const auto x_tgt = Kokkos::subview(tgt_x, i, Kokkos::ALL);
      const auto x_src = Kokkos::subview(src_x, j, Kokkos::ALL);
      const Real xij[2] = {(x_tgt[0] - x_src[0]), (x_tgt[1] - x_src[1])};
      lap += (tgt_values(j) - src_values(i)) * area(j) * kernels.laplacian(xij) / kernels.epssq;
    }
  }
};

template <typename ReducerType>
struct DirectSum {
  using kernel_type = typename ReducerType::kernel_type;
  using value_type = typename ReducerType::value_type;
  using view_type = typename ReducerType::view_type;
  using crd_view = PlaneGeometry::crd_view_type;
  using vec_view = PlaneGeometry::vec_view_type;
  view_type output_view;
  scalar_view_type tgt_f;
  crd_view tgt_x;
  crd_view src_x;
  scalar_view_type src_f;
  kernel_type kernels;
  scalar_view_type area;
  mask_view_type src_mask;
  Index n_src;

  DirectSum(view_type output_view, const crd_view tx, const scalar_view_type tf,
    const crd_view sx, const scalar_view_type f,
    const kernel_type& kernels,
    const scalar_view_type area, const mask_view_type mask, const Index n) :
    output_view(output_view),
    tgt_x(tx),
    tgt_f(tf),
    src_x(sx),
    src_f(f),
    kernels(kernels),
    area(area),
    src_mask(mask),
    n_src(n) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const member_type& thread_team) const {
    const Index i = thread_team.league_rank();

    value_type f_sum;
    Kokkos::parallel_reduce(Kokkos::TeamVectorRange(thread_team, n_src),
      ReducerType(tgt_x, tgt_f, src_x, src_f, kernels, area, src_mask, i), f_sum);
    if constexpr (ReducerType::ndim == 2) {
      output_view(i,0) = f_sum[0];
      output_view(i,1) = f_sum[1];
    }
    else {
      output_view(i) = f_sum;
    }

  }
};

template <typename KernelType>
struct VelocityDirectSum {
  using crd_view = PlaneGeometry::crd_view_type;
  using vec_view = PlaneGeometry::vec_view_type;
  vec_view tgt_u;
  scalar_view_type tgt_double_dot;
  scalar_view_type du1dx1;
  scalar_view_type du1dx2;
  scalar_view_type du2dx1;
  scalar_view_type du2dx2;
  crd_view tgt_x;
  crd_view src_x;
  KernelType kernels;
  scalar_view_type vorticity;
  scalar_view_type divergence;
  scalar_view_type area;
  mask_view_type src_mask;
  Index n_src;

  VelocityDirectSum(vec_view tgt_u, scalar_view_type tgt_dd,
    scalar_view_type du1dx1, scalar_view_type du1dx2, scalar_view_type du2dx1, scalar_view_type du2dx2,
    const crd_view tgt_x, const crd_view src_x,
    const KernelType kernels, const scalar_view_type zeta, const scalar_view_type delta,
    const scalar_view_type area, const mask_view_type mask, const Index n) :
    tgt_u(tgt_u),
    tgt_double_dot(tgt_dd),
    du1dx1(du1dx1),
    du1dx2(du1dx2),
    du2dx1(du2dx1),
    du2dx2(du2dx2),
    tgt_x(tgt_x),
    src_x(src_x),
    kernels(kernels),
    vorticity(zeta),
    divergence(delta),
    area(area),
    src_mask(mask),
    n_src(n) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const member_type& thread_team) const {
    const Index i = thread_team.league_rank();

    Kokkos::Tuple<Real,6> u_sum;
    Kokkos::parallel_reduce(Kokkos::TeamVectorRange(thread_team, n_src),
      PlaneVelocityDirectSumReducer(tgt_x, src_x, kernels, vorticity,
        divergence, area, src_mask, i), u_sum);

    tgt_u(i,0) = u_sum[0];
    tgt_u(i,1) = u_sum[1];
    tgt_double_dot(i) = square(u_sum[2]) + 2*u_sum[3]*u_sum[4] + square(u_sum[5]);
    du1dx1(i) = u_sum[2];
    du1dx2(i) = u_sum[3];
    du2dx1(i) = u_sum[4];
    du2dx2(i) = u_sum[5];
  }
};

template <typename KernelType>
struct PlaneInteriorGradientReducer {
  using crd_view = PlaneGeometry::crd_view_type;
  using vec_view = PlaneGeometry::vec_view_type;
  using value_type = Kokkos::Tuple<Real,2>;
  using kernel_type = KernelType;
  using view_type = vec_view;
  static constexpr Int ndim = 2;
  crd_view tgt_x;
  scalar_view_type tgt_values;
  crd_view src_x;
  KernelType kernels;
  scalar_view_type src_values;
  scalar_view_type area;
  mask_view_type src_mask;
  Index i;
  bool tgt_in_boundary_zone;

  KOKKOS_INLINE_FUNCTION
  PlaneInteriorGradientReducer(const crd_view tgt_x, const scalar_view_type tgt_values,
    const crd_view src_x, const scalar_view_type src_values,
    const KernelType& kernels, const scalar_view_type area,
    const mask_view_type src_mask, const Index i, const bool in_boundary) :
    tgt_x(tgt_x),
    tgt_values(tgt_values),
    src_x(src_x),
    src_values(src_values),
    kernels(kernels),
    area(area),
    src_mask(src_mask),
    i(i),
    tgt_in_boundary_zone(in_boundary) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index& j, value_type& grad) const {
    if (!src_mask(j)) {
      const auto x_tgt = Kokkos::subview(tgt_x, i, Kokkos::ALL);
      const auto x_src = Kokkos::subview(src_x, j, Kokkos::ALL);
      const Real xij[2] = {x_tgt[0] - x_src[0], x_tgt[1] - x_src[1]};
      const Real vij = area(j) * (tgt_values(j) + src_values(i));
      if (tgt_in_boundary_zone) {
        const bool is_interior = PlaneGeometry::norm2(x_tgt) > PlaneGeometry::norm2(x_src);
        if (is_interior) {
          grad[0] += vij * kernels.left_x0_derivative(xij) / kernels.eps;
          grad[1] += vij * kernels.left_x1_derivative(xij) / kernels.eps;
        }
      }
      else {
        grad[0] += vij * kernels.x0_derivative(xij) / kernels.eps;
        grad[1] += vij * kernels.x1_derivative(xij) / kernels.eps;
      }
    }
  }
};

template <typename ReducerType>
struct InteriorRadialGradientDirectSum {
  using kernel_type = typename ReducerType::kernel_type;
  using value_type = typename ReducerType::value_type;
  using view_type = typename ReducerType::view_type;
  using crd_view = PlaneGeometry::crd_view_type;
  using vec_view = PlaneGeometry::vec_view_type;
  view_type output_view;
  scalar_view_type tgt_f;
  crd_view tgt_x;
  crd_view src_x;
  scalar_view_type src_f;
  kernel_type kernels;
  scalar_view_type area;
  mask_view_type src_mask;
  Index n_src;
  Real boundary_radius;

  InteriorRadialGradientDirectSum(view_type output_view, const crd_view tx, const scalar_view_type tf,
    const crd_view sx, const scalar_view_type f,
    const kernel_type& kernels,
    const scalar_view_type area, const mask_view_type mask, const Index n, const Real boundary_radius) :
    output_view(output_view),
    tgt_x(tx),
    tgt_f(tf),
    src_x(sx),
    src_f(f),
    kernels(kernels),
    area(area),
    src_mask(mask),
    n_src(n),
    boundary_radius(boundary_radius) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const member_type& thread_team) const {
    const Index i = thread_team.league_rank();
    const auto x_i = Kokkos::subview(tgt_x, i, Kokkos::ALL);
    const bool in_boundary = PlaneGeometry::norm2(x_i) > square(boundary_radius);
    value_type f_sum;
    Kokkos::parallel_reduce(Kokkos::TeamVectorRange(thread_team, n_src),
      PlaneInteriorGradientReducer(tgt_x, tgt_f, src_x, src_f, kernels, area, src_mask, i, in_boundary), f_sum);
    output_view(i,0) = f_sum[0];
    output_view(i,1) = f_sum[1];
  }
};

} // namespace colloc
} // namespace Lpm
#endif
