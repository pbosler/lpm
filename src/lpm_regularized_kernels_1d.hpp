#ifndef LPM_REGULARIZED_KERNELS_1D_HPP
#define LPM_REGULARIZED_KERNELS_1D_HPP

#include "LpmConfig.h"
#include "lpm_constants.hpp"
#include "lpm_geometry.hpp"
#include "util/lpm_math.hpp"
#include "util/lpm_tuple.hpp"

namespace Lpm {
namespace colloc {

namespace impl {

template <typename KernelType> KOKKOS_INLINE_FUNCTION
Real line_scaled_blob(const Real& x, const KernelType& kernels) {
  const Real xscaled = x/kernels.eps;
  return kernels.blob(xscaled)/kernels.eps;
}

} // namespace impl

/**  Each kernel set provides "blob" functions for interpolation,
  velocity kernels, and derivatives such as the laplacian, all
  with the same order.
*/

struct Line2ndOrder {
  Real eps;
  Real epssq;
  Real denom;
  static constexpr Int order = 2;

  KOKKOS_INLINE_FUNCTION
  explicit Line2ndOrder(const Real eps) : eps(eps), epssq(square(eps)),
    denom(sqrt(constants::PI)) {}

  KOKKOS_INLINE_FUNCTION
  Real blob(const Real& x) const {
    return exp(-square(x))/denom;
  }

  KOKKOS_INLINE_FUNCTION
  Real scaled_blob(const Real& x) const {
    return impl::line_scaled_blob(x, *this);
  }

  KOKKOS_INLINE_FUNCTION
  Real x_derivative(const Real& x) const {
    return -2*x*exp(-square(x))/denom;
  }

  KOKKOS_INLINE_FUNCTION
  Real laplacian(const Real& x) const {
    return 4*exp(-square(x))/denom;
  }
};

struct Line4thOrder {
  Real eps;
  Real epssq;
  Real denom;
  static constexpr Int order = 4;

  KOKKOS_INLINE_FUNCTION
  explicit Line4thOrder(const Real eps) : eps(eps), epssq(square(eps)),
    denom(sqrt(constants::PI)) {}

  KOKKOS_INLINE_FUNCTION
  Real blob(const Real& x) const {
    return (1.5 - square(x))*exp(-square(x))/denom;
  }

  KOKKOS_INLINE_FUNCTION
  Real scaled_blob(const Real& x) const {
    return impl::line_scaled_blob(x, *this);
  }

  KOKKOS_INLINE_FUNCTION
  Real x_derivative(const Real& x) const {
    const Real xsq = square(x);
    return x * (-5 + 2*xsq)*exp(-xsq)/denom;
  }

  KOKKOS_INLINE_FUNCTION
  Real laplacian(const Real& x) const {
    const Real xsq = square(x);
    return (10 - 4*xsq)*exp(-xsq)/denom;
  }
};

struct Line6thOrder {
  Real eps;
  Real epssq;
  Real denom;
  static constexpr Int order = 6;

  KOKKOS_INLINE_FUNCTION
  explicit Line6thOrder(const Real eps) : eps(eps), epssq(square(eps)),
    denom(sqrt(constants::PI)) {}

  KOKKOS_INLINE_FUNCTION
  Real blob(const Real& x) const {
    const Real xsq = square(x);
    return (15.0/8 - 2.5*xsq + 0.5*square(xsq))*exp(-square(x))/denom;
  }

  KOKKOS_INLINE_FUNCTION
  Real scaled_blob(const Real& x) const {
    return impl::line_scaled_blob(x, *this);
  }

  KOKKOS_INLINE_FUNCTION
  Real x_derivative(const Real& x) const {
    const Real xsq = square(x);
    return x * (-35.0/4 + 7*xsq - square(xsq))*exp(-xsq)/denom;
  }

  KOKKOS_INLINE_FUNCTION
  Real laplacian(const Real& x) const {
    const Real xsq = square(x);
    return (17.5 - 14*xsq + 2*square(xsq))*exp(-xsq)/denom;
  }
};

struct Line8thOrder {
  Real eps;
  Real epssq;
  Real denom;
  static constexpr Int order = 8;

  KOKKOS_INLINE_FUNCTION
  explicit Line8thOrder(const Real eps) : eps(eps), epssq(square(eps)),
    denom(sqrt(constants::PI)) {}

  KOKKOS_INLINE_FUNCTION
  Real blob(const Real& x) const {
    const Real xsq = square(x);
    const Real x4th = square(xsq);
    return ( 35.0/16 - 35*xsq/8 + 7*x4th/4 - xsq*x4th/6) * exp(-square(x))/denom;
  }

  KOKKOS_INLINE_FUNCTION
  Real scaled_blob(const Real& x) const {
    return impl::line_scaled_blob(x, *this);
  }

  KOKKOS_INLINE_FUNCTION
  Real x_derivative(const Real& x) const {
    const Real xsq = square(x);
    const Real x4th = square(xsq);
    return x * (-105.0/4 + 63*xsq/4 -4.5*x4th + xsq*x4th/6)*exp(-xsq)/denom;
  }

  KOKKOS_INLINE_FUNCTION
  Real laplacian(const Real& x) const {
    const Real xsq = square(x);
    const Real x4th = square(xsq);
    return (105.0/4 - 63.0/2*xsq + 9*x4th - 2*xsq*x4th/3)*exp(-xsq)/denom;
  }
};

template <typename KernelType>
struct LineXDerivativeReducer {
  using crd_view = scalar_view_type;
  using vec_view = scalar_view_type;
  using value_type = Real;
  using kernel_type = KernelType;
  static constexpr Int kernel_order = KernelType::order;
  using view_type = scalar_view_type;
  static constexpr Int ndim = 1;
  static constexpr Int order = 1;
  scalar_view_type x;
  scalar_view_type f_values;
  scalar_view_type length;
  kernel_type kernels;
  Index i;

  KOKKOS_INLINE_FUNCTION
  LineXDerivativeReducer(const scalar_view_type x, const scalar_view_type f, const KernelType& kernels,
    const scalar_view_type l, const Index i) :
    x(x), f_values(f), length(l), kernels(kernels), i(i) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index& j, Real& dfdx) const {
     const Real xij = (x(i) - x(j))/kernels.eps;
     const Real Fi = f_values(i) * length(i);
     const Real Fj = f_values(j) * length(j);
     dfdx += (Fi * length(j) + Fj * length(i)) * kernels.x_derivative(xij) / kernels.eps * length(j);
  }
};

template <typename KernelType>
struct Line2ndDerivativeReducer {
  using crd_view = scalar_view_type;
  using vec_view = scalar_view_type;
  using value_type = Real;
  using kernel_type = KernelType;
  static constexpr Int kernel_order = KernelType::order;
  using view_type = scalar_view_type;
  static constexpr Int ndim = 1;
  static constexpr Int order = 2;
  scalar_view_type x;
  scalar_view_type f_values;
  scalar_view_type length;
  kernel_type kernels;
  Index i;

  KOKKOS_INLINE_FUNCTION
  Line2ndDerivativeReducer(const scalar_view_type x, const scalar_view_type f, const KernelType& kernels,
    const scalar_view_type l, const Index i) :
    x(x),
    f_values(f),
    kernels(kernels),
    length(l),
    i(i) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index& j, Real& df2dx2) const {
    const Real xij_scal = (x(i) - x(j)) / kernels.eps;
    const Real kscal = kernels.laplacian(xij_scal) / kernels.eps;
    df2dx2 += (f_values(j) - f_values(i)) * length(j) * kscal;
//     const Real Fi = f_values(i) * length(i);
//     const Real Fj = f_values(j) * length(j);
//     df2dx2 += (Fi*length(j) - Fj*length(i)) * kscal;
  }
};

template <typename ReducerType>
struct LineDirectSum {
  using kernel_type = typename ReducerType::kernel_type;
  using value_type = typename ReducerType::value_type;
  using view_type = typename ReducerType::view_type;
  static constexpr Int kernel_order = ReducerType::kernel_order;
  static constexpr Int derivative_order = ReducerType::order;
  view_type output_view;
  scalar_view_type x;
  scalar_view_type f;
  kernel_type kernels;
  scalar_view_type length;
  Index n_src;

  LineDirectSum(scalar_view_type output_view, const scalar_view_type x,
    const scalar_view_type f, const kernel_type& kernels, const scalar_view_type length,
    const Index n) :
    output_view(output_view),
    x(x), f(f),
    kernels(kernels),
    length(length),
    n_src(n) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const member_type& thread_team) const {
    const Index i = thread_team.league_rank();
    Real output_value;
    Kokkos::parallel_reduce(Kokkos::TeamVectorRange(thread_team, n_src),
      ReducerType(x, f, kernels, length, i), output_value);
    output_view(i) = output_value / pow(kernels.eps, derivative_order);
  }
};

} // namespace colloc
} // namespace Lpm
#endif
