#ifndef LPM_HIGH_ORDER_SWE_HPP
#define LPM_HIGH_ORDER_SWE_HPP

#include "LpmConfig.h"
#include "lpm_constants.hpp"
#include "lpm_geometry.hpp"
#include "util/lpm_math.hpp"
#include "util/lpm_tuple.hpp"

namespace Lpm {

struct Blob2ndOrderPlane {
  using geo = PlaneGeometry;

  KOKKOS_INLINE_FUNCTION
  static Real value(const Real& r) { return exp(-square(r)) / constants::PI; }

  template <typename XType>
  KOKKOS_INLINE_FUNCTION static Real scaled_value(const XType& x,
                                                  const Real& eps) {
    const Real scaled_r = PlaneGeometry::mag(x) / eps;
    return value(scaled_r) / square(eps);
  }

  KOKKOS_INLINE_FUNCTION
  static Real qfn(const Real& r, const Real& eps) {
    return 1 - exp(-square(r) / square(eps));
  }

  template <typename XType>
  KOKKOS_INLINE_FUNCTION static Real qfn(const XType& x, const Real& eps) {
    const Real r = PlaneGeometry::mag(x);
    return qfn(r, eps);
  }
};

struct Blob8thOrderPlane {
  using geo = PlaneGeometry;

  KOKKOS_INLINE_FUNCTION
  static Real value(const Real& r) {
    const Real rsq   = square(r);
    const Real coeff = 4 - 6 * rsq + 2 * square(rsq) - rsq * square(rsq) / 6;
    return coeff * exp(-rsq) / constants::PI;
  }

  template <typename XType>
  KOKKOS_INLINE_FUNCTION static Real scaled_value(const XType& x,
                                                  const Real& eps) {
    const Real scaled_r = PlaneGeometry::mag(x) / eps;
    return value(scaled_r) / square(eps);
  }

  KOKKOS_INLINE_FUNCTION
  static Real qfn(const Real& r, const Real& eps) {
    const Real rsq   = square(r);
    const Real epssq = square(eps);
    const Real coeff = (6 - 18 * rsq / epssq + 9 * square(rsq) / square(epssq) -
                        rsq * square(rsq) / (epssq * square(epssq))) /
                       6;
    return 1 - coeff * exp(-rsq / epssq);
  }

  template <typename XType>
  KOKKOS_INLINE_FUNCTION static Real qfn(const XType& x, const Real& eps) {
    const Real r = PlaneGeometry::mag(x);
    return qfn(r, eps);
  }
};

template <typename BlobType>
struct HighOrderSWEKernels {
  using geo = PlaneGeometry;

  template <typename VecType>
  KOKKOS_INLINE_FUNCTION static Real velocity_prefactor(const VecType& xin,
                                                        const Real& eps) {
    const Real q_argin = PlaneGeometry::mag(xin) / eps;
    return BlobType::qfn(q_argin) /
           (2 * constants::PI * PlaneGeometry::norm2(xin));
  }

  template <typename XType, typename YType>
  KOKKOS_INLINE_FUNCTION static Kokkos::Tuple<Real, 2> kzeta(const XType& x,
                                                             const YType& y,
                                                             const Real& eps) {
    Kokkos::Tuple<Real, 2> result;
    const Real x_minus_y[2] = {x[0] - y[0], x[1] - y[1]};
    const Real coeff        = velocity_prefactor(x_minus_y, eps);
    result[0]               = -x_minus_y[1] * coeff;
    result[1]               = x_minus_y[0] * coeff;
    return result;
  }

  template <typename XType, typename YType>
  KOKKOS_INLINE_FUNCTION static Kokkos::Tuple<Real, 2> ksigma(const XType& x,
                                                              const YType& y,
                                                              const Real& eps) {
    Kokkos::Tuple<Real, 2> result;
    result[0]        = x[0] - y[0];
    result[1]        = x[1] - y[1];
    const Real coeff = velocity_prefactor(result, eps);
    PlaneGeometry::scale(coeff, result);
    return result;
  }

  template <typename VecType>
  KOKKOS_INLINE_FUNCTION static Real Rij(const VecType& x, const Real& eps,
                                         const int& i, const int& j) {
    LPM_KERNEL_ASSERT(i == 0 or i == 1);
    LPM_KERNEL_ASSERT(j == 0 or j == 1);

    const Real rsq     = PlaneGeometry::norm2(x);
    const Real q_argin = PlaneGeometry::mag(x) / eps;
    const Real qval    = BlobType::qfn(q_argin);

    const Real coeff =
        qval / (constants::PI * rsq) - BlobType::scaled_value(x, eps);
    Real result = coeff * x[i] * x[j] / rsq;
    if (i == j) {
      result -= qval / (2 * constants::PI * rsq);
    }
    return result;
  }

  template <typename XType, typename YType>
  KOKKOS_INLINE_FUNCTION static Kokkos::Tuple<Real, 4> grad_kzeta(
      const XType& x, const YType& y, const Real& eps) {
    Kokkos::Tuple<Real, 4> result;
    const Real x_minus_y[2] = {x[0] - y[0], x[1] - y[1]};
    result[0]               = Rij(x_minus_y, eps, 1, 0);
    result[1]               = Rij(x_minus_y, eps, 1, 1);
    result[2]               = -Rij(x_minus_y, eps, 0, 0);
    result[3]               = -Rij(x_minus_y, eps, 0, 1);
    return result;
  }

  template <typename XType, typename YType>
  KOKKOS_INLINE_FUNCTION static Kokkos::Tuple<Real, 4> grad_ksigma(
      const XType& x, const YType& y, const Real& eps) {
    Kokkos::Tuple<Real, 4> result;
    const Real x_minus_y[2] = {x[0] - y[0], x[1] - y[1]};
    result[0]               = -Rij(x_minus_y, eps, 0, 0);
    result[1]               = -Rij(x_minus_y, eps, 0, 1);
    result[2]               = -Rij(x_minus_y, eps, 1, 0);
    result[3]               = -Rij(x_minus_y, eps, 1, 1);
    return result;
  }
};

}  // namespace Lpm

#endif
