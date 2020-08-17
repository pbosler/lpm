#ifndef LPM_PSE_HPP
#define LPM_PSE_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmUtilities.hpp"
#include "LpmKokkosUtil.hpp"

#include <cassert>

#include <cmath>

namespace Lpm {

KOKKOS_INLINE_FUNCTION
static Real pse_eps(const Real& dx, const Real& p=17.0/20.0 /* default value p = 0.85 */) {
  assert(p<1);
  return std::pow(dx,p);
}

template <typename Geo, typename CV> KOKKOS_INLINE_FUNCTION
static Real pse_kernel_input(const CV& x0, const CV& x1, const Real& eps) {
  return Geo::distance(x0, x1)/eps;
}

KOKKOS_INLINE_FUNCTION
static Real bivariateDeltaOrder8(const Real& r) {
  const Real rsq = square(r);
  const Real exp_fac = std::exp(-rsq)/PI;
  const Real pre_fac = 4 - 6*rsq + 2*square(rsq) - rsq*square(rsq)/6;
  return pre_fac*exp_fac;
}

KOKKOS_INLINE_FUNCTION
static Real bivariateLaplacianOrder8(const Real& r) {
  const Real rsq = square(r);
  const Real exp_fac = std::exp(-rsq)/PI;
  const Real pre_fac = 40*(1-rsq) + 10*square(rsq) - 2*rsq*square(rsq)/3;
  return pre_fac*exp_fac;
}

struct PlanePSEDelta8Reduce {
  typedef Real value_type;
  typedef typename PlaneGeometry::crd_view_type crd_view;
  Index tgt_ind;
  crd_view tgtx;
  crd_view srcx;
  scalar_view_type srcdata;
  scalar_view_type srcarea;
  mask_view_type srcmask;
  Real pse_eps;
  Real eps2;
  Index i;

  KOKKOS_INLINE_FUNCTION
  PlanePSEDelta8Reduce(const Index& ind, const crd_view& tgts, const crd_view& srcs,
    const scalar_view_type& srcvals, const scalar_view_type& a, const mask_view_type sm,
    const Real& ep): i(ind),
    tgt_ind(i), tgtx(tgts), srcx(srcs), srcdata(srcvals), srcarea(a), srcmask(sm),
    pse_eps(ep), eps2(ep*ep) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index& j, value_type& f) const {
    Real fj = 0;
    if (!srcmask(j)) {
      const auto mytgt = ko::subview(tgtx, i, ko::ALL());
      const auto mysrc = ko::subview(srcx, j, ko::ALL());
      const Real rscaled = PlaneGeometry::distance(mytgt,mysrc)/pse_eps;
      fj = srcdata(j)*srcarea(j)*bivariateDeltaOrder8(rscaled)/eps2;
    }
    f += fj;
  }
};

struct PlanePSELaplacian8Reduce {
  typedef Real value_type;
  typedef typename PlaneGeometry::crd_view_type crd_view;
  Index tgt_ind;
  crd_view tgtx;
  scalar_view_type tgtf;
  crd_view srcx;
  scalar_view_type srcf;
  scalar_view_type srcarea;
  Real eps;

  KOKKOS_INLINE_FUNCTION
  PlanePSELaplacian8Reduce(const Index& i, const crd_view& tx, const scalar_view_type& tf,
    const crd_view& sx, const scalar_view_type& sf, const scalar_view_type& sa,
    const Real& pe) : tgt_ind(i), tgtx(tx), tgtf(tf), srcx(sx), srcf(sf), srcarea(sa),
    eps(pe) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index& j, value_type& lap) const {
    const auto mtgt = ko::subview(tgtx, tgt_ind, ko::ALL);
    const auto msrc = ko::subview(srcx, j, ko::ALL);
    const Real rscl = PlaneGeometry::distance(mtgt,msrc)/eps;
    const Real kern = bivariateLaplacianOrder8(rscl)/square(eps);
    const Real val = (srcf(j)-tgtf(tgt_ind))*kern*srcarea(j);
    lap += val;
  }
};

struct PlanePSELaplacian {
  typedef typename PlaneGeometry::crd_view_type crd_view;
  scalar_view_type laplacian;
  crd_view tgtx;
  scalar_view_type tgtf;
  crd_view srcx;
  scalar_view_type srcf;
  scalar_view_type srcarea;
  Real eps;
  Index nsrc;

  PlanePSELaplacian(scalar_view_type& lap_out, const crd_view& tx, const scalar_view_type& tf,
    const crd_view& sx, const scalar_view_type& sf, const scalar_view_type& sa,
    const Real& pe, const Index& ns) : laplacian(lap_out), tgtx(tx), tgtf(tf), srcx(sx), srcf(sf),
    srcarea(sa), eps(pe), nsrc(ns) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const member_type& mbr) const {
    const Index i = mbr.league_rank();
    Real lap=0;
    ko::parallel_reduce(ko::TeamThreadRange(mbr, nsrc),
      PlanePSELaplacian8Reduce(i, tgtx, tgtf, srcx, srcf, srcarea, eps), lap);
    laplacian(i) = lap/square(eps);
  }
};


struct PlanePSEScalarInterp {
  scalar_view_type finterp;
  typedef typename PlaneGeometry::crd_view_type crd_view;
  crd_view tgtx;
  crd_view srcx;
  scalar_view_type srcdata;
  scalar_view_type srcarea;
  mask_view_type srcmask;
  Real eps;
  Index nsrc;

  PlanePSEScalarInterp(scalar_view_type& f, const crd_view& t, const crd_view& s,
    const scalar_view_type& fs, const scalar_view_type& sa, const mask_view_type& sm,
    const Real& ep, const Index& n) : finterp(f), tgtx(t), srcx(s), srcdata(fs),
    srcarea(sa), srcmask(sm), eps(ep), nsrc(n) {}

  KOKKOS_INLINE_FUNCTION
  void operator () (const member_type& mbr) const {
    const Index i = mbr.league_rank();
    Real f;
    ko::parallel_reduce(ko::TeamThreadRange(mbr, nsrc),
      PlanePSEDelta8Reduce(i, tgtx, srcx, srcdata, srcarea, srcmask, eps), f);
    finterp(i) = f;
  }
};

}
#endif
