#ifndef LPM_PSE_HPP
#define LPM_PSE_HPP

#include "LpmConfig.h"
#include "lpm_assert.hpp"
#include "lpm_constants.hpp"
#include "lpm_geometry.hpp"
#include "lpm_kokkos_defs.hpp"
#include "util/lpm_math.hpp"
#include "util/lpm_tuple.hpp"

namespace Lpm {
namespace pse {

template <typename Geo>
struct PSEKernel {
  static constexpr Int ndim = Geo::ndim;

  KOKKOS_INLINE_FUNCTION
  static Real get_epsilon(const Real dx, const Real p = 11.0 / 20) {
    LPM_KERNEL_ASSERT(p < 1);
    return pow(dx, p);
  }

  template <typename CV>
  KOKKOS_INLINE_FUNCTION static Real kernel_input(const CV& x, const Real eps) {
    return Geo::mag(x) / eps;
  }

  template <typename CV, typename CV2>
  KOKKOS_INLINE_FUNCTION static Real kernel_input(const CV x0, const CV2 x1,
                                                  const Real& eps) {
    return Geo::distance(x0, x1) / eps;
  }
};

struct BivariateOrder8 {
  using geo                 = PlaneGeometry;
  static constexpr Int ndim = geo::ndim;

  template <typename VecType1, typename VecType2>
  KOKKOS_INLINE_FUNCTION static Real kernel_input(const VecType1& x1,
                                                  const VecType2& x2,
                                                  const Real& eps) {
    return PSEKernel<geo>::kernel_input(x1, x2, eps);
  }

  KOKKOS_INLINE_FUNCTION
  static Real delta(const Real r) {
    const Real rsq     = square(r);
    const Real exp_fac = exp(-rsq) / constants::PI;
    const Real pre_fac = 4 - 6 * rsq + 2 * square(rsq) - rsq * square(rsq) / 6;
    return pre_fac * exp_fac;
  }

  KOKKOS_INLINE_FUNCTION
  static Real first_derivative(const Real xi, const Real r) {
    const Real rsq = square(r);
    const Real pre_fac =
        20 * (rsq - 1) - 5 * square(rsq) + rsq * square(rsq) / 3;
    const Real exp_fac = xi * exp(-rsq) / constants::PI;
    return pre_fac * exp_fac;
  }

  KOKKOS_INLINE_FUNCTION
  static Real laplacian(const Real r) {
    const Real rsq     = square(r);
    const Real exp_fac = exp(-rsq) / constants::PI;
    const Real pre_fac =
        40 * (1 - rsq) + 10 * square(rsq) - 2 * rsq * square(rsq) / 3;
    return pre_fac * exp_fac;
  }
};

template <typename Geo>
struct BivariateOrder2 {
  using geo                 = PlaneGeometry;
  static constexpr Int ndim = geo::ndim;

  KOKKOS_INLINE_FUNCTION
  static Real delta(const Real r) {
    const Real rsq     = square(r);
    const Real exp_fac = exp(-rsq) / constants::PI;
    const Real pre_fac = 1;
    return pre_fac * exp_fac;
  }

  KOKKOS_INLINE_FUNCTION
  static Real first_derivative(const Real xi, const Real r) {
    const Real rsq     = square(r);
    const Real exp_fac = xi * exp(-rsq) / constants::PI;
    const Real pre_fac = -2;
    return pre_fac * exp_fac;
  }

  KOKKOS_INLINE_FUNCTION
  static Real left_first_derivative(const Real xi, const Real r) {
    const Real rsq     = square(r);
    const Real exp_fac = xi * exp(-rsq) / constants::PI;
    const Real pre_fac = -20 + 8 * rsq;
    return pre_fac * exp_fac;
  }

  KOKKOS_INLINE_FUNCTION
  static Real laplacian(const Real r) {
    const Real rsq     = square(r);
    const Real exp_fac = exp(-rsq) / constants::PI;
    const Real pre_fac = 4;
    return pre_fac * exp_fac;
  }
};

template <typename PSEKind>
struct DeltaReducer {
  typedef Real value_type;
  using geo                 = typename PSEKind::geo;
  static constexpr Int ndim = PSEKind::ndim;
  using crd_view            = typename geo::crd_view_type;
  Index tgt_idx;
  crd_view tgtx;
  crd_view srcx;
  scalar_view_type src_data;
  scalar_view_type src_area;
  Real epsilon;
  Real eps_sq;

  KOKKOS_INLINE_FUNCTION
  DeltaReducer(Index tidx, const crd_view targets, const crd_view sources,
               const scalar_view_type src_vals, const scalar_view_type area,
               Real ep)
      : tgt_idx(tidx),
        tgtx(targets),
        srcx(sources),
        src_data(src_vals),
        src_area(area),
        epsilon(ep) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index j, Real& val) const {
    const auto mtgt = Kokkos::subview(tgtx, tgt_idx, Kokkos::ALL);
    const auto msrc = Kokkos::subview(srcx, j, Kokkos::ALL);
    if constexpr (std::is_same<geo, SphereGeometry>::value) {
      if (geo::dot(mtgt, msrc) > 0) {
        const auto rscaled = PSEKernel<geo>::kernel_input(mtgt, msrc, epsilon);
        val += src_data(j) * src_area(j) * PSEKind::delta(rscaled) /
               square(epsilon);
      }
    } else {
      const auto rscaled = PSEKernel<geo>::kernel_input(mtgt, msrc, epsilon);
      val +=
          src_data(j) * src_area(j) * PSEKind::delta(rscaled) / square(epsilon);
      //            (ndim == 2 ? square(epsilon) : cube(epsilon));
    }
  }
};

template <typename PSEKind>
struct LaplacianReducer {
  typedef Real value_type;
  static constexpr Int ndim = PSEKind::ndim;
  using geo                 = typename PSEKind::geo;
  using crd_view            = typename geo::crd_view_type;
  Index tgt_idx;
  crd_view tgtx;
  crd_view srcx;
  scalar_view_type tgt_data;
  scalar_view_type src_data;
  scalar_view_type src_area;
  Real epsilon;
  Real eps_sq;

  KOKKOS_INLINE_FUNCTION
  LaplacianReducer(const Index tidx, const crd_view targets,
                   const crd_view sources, const scalar_view_type target_vals,
                   const scalar_view_type source_vals,
                   const scalar_view_type source_area, const Real eps)
      : tgt_idx(tidx),
        tgtx(targets),
        srcx(sources),
        tgt_data(target_vals),
        src_data(source_vals),
        src_area(source_area),
        epsilon(eps),
        eps_sq(square(eps)) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index j, Real& val) const {
    const auto mtgt    = Kokkos::subview(tgtx, tgt_idx, Kokkos::ALL);
    const auto msrc    = Kokkos::subview(srcx, j, Kokkos::ALL);
    const auto rscaled = PSEKernel<geo>::kernel_input(mtgt, msrc, epsilon);
    if constexpr (std::is_same<geo, SphereGeometry>::value) {
      if (geo::dot(mtgt, msrc) > 0) {
        const auto kern = PSEKind::laplacian(rscaled) /
                          eps_sq;  //(ndim == 2 ? eps_sq : cube(epsilon));
        val += (src_data(j) - tgt_data(tgt_idx)) * src_area(j) * kern / eps_sq;
      }
    } else {
      const auto kern = PSEKind::laplacian(rscaled) /
                        eps_sq;  //(ndim == 2 ? eps_sq : cube(epsilon));
      val += (src_data(j) - tgt_data(tgt_idx)) * src_area(j) * kern / eps_sq;
    }
  }
};

template <typename PSEKind>
struct ScalarInterpolation {
  scalar_view_type finterp;
  using geo      = typename PSEKind::geo;
  using crd_view = typename geo::crd_view_type;
  crd_view tgtx;
  crd_view srcx;
  scalar_view_type src_data;
  scalar_view_type src_area;
  Real epsilon;
  Index nsrc;

  ScalarInterpolation(scalar_view_type fout, crd_view tx, crd_view sx,
                      scalar_view_type srcf, scalar_view_type srca, Real eps,
                      Index ns)
      : finterp(fout),
        tgtx(tx),
        srcx(sx),
        src_data(srcf),
        src_area(srca),
        epsilon(eps),
        nsrc(ns) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type& mbr) const {
    const Index i = mbr.league_rank();
    Real f;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(mbr, nsrc),
        DeltaReducer<PSEKind>(i, tgtx, srcx, src_data, src_area, epsilon), f);
    finterp(i) = f;
  }
};

template <typename PSEKind>
struct ScalarLaplacian {
  using geo      = typename PSEKind::geo;
  using crd_view = typename geo::crd_view_type;
  scalar_view_type flaplacian;
  crd_view tgtx;
  crd_view srcx;
  scalar_view_type tgtf;
  scalar_view_type srcf;
  scalar_view_type src_area;
  Real epsilon;
  Index nsrc;

  ScalarLaplacian(scalar_view_type flap, crd_view tx, crd_view sx,
                  scalar_view_type tf, scalar_view_type sf, scalar_view_type sa,
                  Real eps, Index ns)
      : flaplacian(flap),
        tgtx(tx),
        srcx(sx),
        tgtf(tf),
        srcf(sf),
        src_area(sa),
        epsilon(eps),
        nsrc(ns) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type mbr) const {
    const Index i = mbr.league_rank();
    Real lap;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(mbr, nsrc),
        LaplacianReducer<PSEKind>(i, tgtx, srcx, tgtf, srcf, src_area, epsilon),
        lap);
    flaplacian(i) = lap;
  }
};

}  // namespace pse
}  // namespace Lpm

#endif
