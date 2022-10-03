#ifndef LPM_PSE_HPP
#define LPM_PSE_HPP

#include "LpmConfig.h"
#include "lpm_assert.hpp"
#include "lpm_geometry.hpp"
#include "lpm_constants.hpp"
#include "lpm_kokkos_defs.hpp"
#include "util/lpm_tuple.hpp"
#include "util/lpm_math.hpp"

namespace Lpm {
namespace pse {

template <typename Geo> struct PSEKernel {
  static constexpr Int ndim = Geo::ndim;

  KOKKOS_INLINE_FUNCTION
  static Real epsilon(const Real dx, const Real p=17.0/20) {
    LPM_KERNEL_ASSERT(p<1);
    return pow(dx,p);
  }

  template <typename CV, typename CV2> KOKKOS_INLINE_FUNCTION
  static Real kernel_input(const CV x0, const CV2 x1, const Real& eps) {
    return Geo::distance(x0,x1)/eps;
  }
};


struct BivariateOrder8 : public PSEKernel<PlaneGeometry> {
  typedef PlaneGeometry geo;
  static constexpr Int ndim = 2;

  KOKKOS_INLINE_FUNCTION
  static Real delta(const Real r) {
    const Real rsq = square(r);
    const Real exp_fac = exp(-rsq)/constants::PI;
    const Real pre_fac = 4 - 6*rsq + 2*square(rsq) - rsq*square(rsq)/6;
    return pre_fac * exp_fac;
  }

  KOKKOS_INLINE_FUNCTION
  static Real laplacian(const Real r) {
    const Real rsq = square(r);
    const Real exp_fac = exp(-rsq)/constants::PI;
    const Real pre_fac = 40*(1-rsq) + 10*square(rsq) -2*rsq*square(rsq)/3;
    return pre_fac * exp_fac;
  }
};

template <typename PSEKind> struct DeltaReducer {
  typedef Real value_type;
  typedef typename PSEKind::geo::crd_view_type crd_view;
  static constexpr Int ndim = PSEKind::ndim;
  Index tgt_idx;
  crd_view tgtx;
  crd_view srcx;
  scalar_view_type src_data;
  scalar_view_type src_area;
  Real epsilon;
  Real eps_sq;

  KOKKOS_INLINE_FUNCTION
  DeltaReducer(Index tidx, const crd_view targets, const crd_view sources,
    const scalar_view_type src_vals, const scalar_view_type area, Real ep) :
    tgt_idx(tidx),
    tgtx(targets),
    srcx(sources),
    src_data(src_vals),
    src_area(area),
    epsilon(ep) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index j, Real& val) const {
    const auto mtgt = Kokkos::subview(tgtx, tgt_idx, Kokkos::ALL);
    const auto msrc = Kokkos::subview(srcx, j, Kokkos::ALL);
    const auto rscaled = PSEKind::kernel_input(mtgt, msrc, epsilon);
    val += src_data(j) * src_area(j) * PSEKind::delta(rscaled) /
        (ndim == 2 ? square(epsilon) : cube(epsilon));
  }
};

template <typename PSEKind> struct LaplacianReducer {
  typedef Real value_type;
  typedef typename PSEKind::geo::crd_view_type crd_view;
  static constexpr Int ndim = PSEKind::ndim;
  Index tgt_idx;
  crd_view tgtx;
  crd_view srcx;
  scalar_view_type tgt_data;
  scalar_view_type src_data;
  scalar_view_type src_area;
  Real epsilon;
  Real eps_sq;

  KOKKOS_INLINE_FUNCTION
  LaplacianReducer(const Index tidx, const crd_view targets, const crd_view sources,
    const scalar_view_type target_vals, const scalar_view_type source_vals,
    const scalar_view_type source_area, const Real eps) :
    tgt_idx(tidx),
    tgtx(targets),
    srcx(sources),
    tgt_data(target_vals),
    src_data(source_vals),
    src_area(source_area),
    epsilon(eps),
    eps_sq(square(eps)) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index j, Real& val) const {
    const auto mtgt = Kokkos::subview(tgtx, tgt_idx, Kokkos::ALL);
    const auto msrc = Kokkos::subview(srcx, j, Kokkos::ALL);
    const auto rscaled = PSEKind::kernel_input(mtgt, msrc, epsilon);
    const auto kern = PSEKind::laplacian(rscaled) /
      (ndim == 2 ? eps_sq : cube(epsilon));
    val += (src_data(j) - tgt_data(tgt_idx))*src_area(j)*kern/eps_sq;
  }
};

template <typename PSEKind> struct ScalarInterpolation {
  scalar_view_type finterp;
  typename PSEKind::geo::crd_view_type tgtx;
  typename PSEKind::geo::crd_view_type srcx;
  scalar_view_type src_data;
  scalar_view_type src_area;
  Real epsilon;
  Index nsrc;

  ScalarInterpolation(scalar_view_type fout,
    typename PSEKind::geo::crd_view_type tx,
    typename PSEKind::geo::crd_view_type sx,
    scalar_view_type srcf,
    scalar_view_type srca,
    Real eps,
    Index ns) :
    finterp(fout),
    tgtx(tx),
    srcx(sx),
    src_data(srcf),
    src_area(srca),
    epsilon(eps),
    nsrc(ns) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const member_type& mbr) const {
    const Index i = mbr.league_rank();
    Real f;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(mbr,nsrc),
      DeltaReducer<PSEKind>(i, tgtx, srcx, src_data, src_area, epsilon), f);
    finterp(i) = f;
  }
};

template <typename PSEKind> struct ScalarLaplacian {
  scalar_view_type flaplacian;
  typename PSEKind::geo::crd_view_type tgtx;
  typename PSEKind::geo::crd_view_type srcx;
  scalar_view_type tgtf;
  scalar_view_type srcf;
  scalar_view_type src_area;
  Real epsilon;
  Index nsrc;

  ScalarLaplacian(scalar_view_type flap,
    typename PSEKind::geo::crd_view_type tx,
    typename PSEKind::geo::crd_view_type sx,
    scalar_view_type tf,
    scalar_view_type sf,
    scalar_view_type sa,
    Real eps,
    Index ns) :
    flaplacian(flap),
    tgtx(tx),
    srcx(sx),
    tgtf(tf),
    srcf(sf),
    src_area(sa),
    epsilon(eps),
    nsrc(ns) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const member_type mbr) const {
    const Index i = mbr.league_rank();
    Real lap;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(mbr, nsrc),
      LaplacianReducer<PSEKind>(i, tgtx, srcx, tgtf, srcf, src_area, epsilon), lap);
      flaplacian(i) = lap;
  }

};

} // namespace pse
} // namespace Lpm

#endif
