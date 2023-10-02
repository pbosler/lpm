#ifndef LPM_SWE_TENDENCIES_HPP
#define LPM_SWE_TENDENCIES_HPP

#include "LpmConfig.h"
#include "lpm_swe_kernels.hpp"

namespace Lpm {

/**
*/
template <typename Geo>
struct SWEPassiveTendencies {
  static constexpr int ndim = Geo::ndim;
  using crd_view = typename Geo::crd_view_type;
  using vec_view = typename Geo::vec_view_type;
  using coriolis_type = Coriolis<Geo>;
  scalar_view_type dzeta;
  scalar_view_type dsigma;
  scalar_view_type ddepth;
  crd_view x;
  vec_view velocity;
  scalar_view_type rel_vort;
  scalar_view_type divergence;
  scalar_view_type depth;
  scalar_view_type double_dot;
  scalar_view_type surface_laplacian;
  coriolis_type coriolis;
  Real g;
  Real dt;

  SWEPassiveTendencies(scalar_view_type dzeta, scalar_view_type dsigma, scalar_view_type ddepth,
    const crd_view x, const vec_view u, const scalar_view_type zeta,
    const scalar_view_type sigma, const scalar_view_type h, const scalar_view_type ddot,
    const scalar_view_type surflap, const coriolis_type c, const Real g, const Real dt) :
    dzeta(dzeta),
    dsigma(dsigma),
    ddepth(ddepth),
    x(x),
    velocity(u),
    rel_vort(zeta),
    divergence(sigma),
    depth(h),
    double_dot(ddot),
    surface_laplacian(surflap),
    coriolis(c),
    g(g),
    dt(dt) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index i) const {
    const auto mcrd = Kokkos::subview(x, i, Kokkos::ALL);
    const auto mvel = Kokkos::subview(velocity, i, Kokkos::ALL);
    const Real coriolis_f = coriolis.f(mcrd[ndim-1]);
    const Real coriolis_dfdt = coriolis.dfdt(mvel[ndim-1]);
    dzeta(i) = dt * (
      - coriolis_dfdt - (rel_vort(i) + coriolis_f)*divergence(i));
    dsigma(i) = dt * (
      -coriolis_f * rel_vort(i) - double_dot(i) - g*surface_laplacian(i));
    ddepth(i) = dt * (-divergence(i) * depth(i));
  }
};

template <typename Geo>
struct SWEActiveTendencies {
  static constexpr int ndim = Geo::ndim;
  using crd_view = typename Geo::crd_view_type;
  using vec_view = typename Geo::vec_view_type;
  using coriolis_type = Coriolis<Geo>;
  scalar_view_type dzeta;
  scalar_view_type dsigma;
  scalar_view_type darea;
  crd_view x;
  vec_view velocity;
  scalar_view_type rel_vort;
  scalar_view_type divergence;
  scalar_view_type area;
  scalar_view_type double_dot;
  scalar_view_type surface_laplacian;
  coriolis_type coriolis;
  Real g;
  Real dt;

  SWEActiveTendencies(scalar_view_type dzeta, scalar_view_type dsigma, scalar_view_type darea,
    const crd_view x, const vec_view u, const scalar_view_type zeta, const scalar_view_type sigma,
    const scalar_view_type a, const scalar_view_type ddot, const scalar_view_type surflap,
    const coriolis_type c, const Real g, const Real dt) :
    dzeta(dzeta),
    dsigma(dsigma),
    darea(darea),
    x(x),
    velocity(u),
    rel_vort(zeta),
    divergence(sigma),
    area(a),
    double_dot(ddot),
    surface_laplacian(surflap),
    coriolis(c),
    g(g),
    dt(dt) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index i) const {
    const auto mcrd = Kokkos::subview(x, i, Kokkos::ALL);
    const auto mvel = Kokkos::subview(velocity, i, Kokkos::ALL);
    const Real coriolis_f = coriolis.f(mcrd[ndim-1]);
    const Real coriolis_dfdt = coriolis.dfdt(mvel[ndim-1]);
    dzeta(i) = dt * (
      - coriolis_dfdt - (rel_vort(i) + coriolis_f)*divergence(i));
    dsigma(i) = dt * (
      -coriolis_f * rel_vort(i) - double_dot(i) - g*surface_laplacian(i));
    darea(i) = dt * (divergence(i) * area(i));
  }
};

template <typename BottomType>
struct SurfaceUpdatePassive {
  using crd_view = Kokkos::View<Real**>;
  scalar_view_type surface_height;
  scalar_view_type depth;
  crd_view x;
  BottomType topo;

  SurfaceUpdatePassive(scalar_view_type surf, const scalar_view_type h,
    const crd_view x,
    const BottomType& bottom) :
    surface_height(surf),
    depth(h),
    x(x),
    topo(bottom) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index i) const {
    const auto mcrd = Kokkos::subview(x, i, Kokkos::ALL);
    surface_height(i) = depth(i) + topo(mcrd);
  }
};

template <typename BottomType>
struct SurfaceUpdateActive {
  using crd_view = Kokkos::View<Real**>;
  scalar_view_type surface_height;
  scalar_view_type depth;
  scalar_view_type mass;
  scalar_view_type area;
  mask_view_type mask;
  crd_view x;
  BottomType topo;

  SurfaceUpdateActive(scalar_view_type surf, scalar_view_type h,
    const scalar_view_type m,
    const scalar_view_type a,
    const mask_view_type mm,
    const crd_view x,
    const BottomType& bottom) :
    surface_height(surf),
    depth(h),
    mass(m),
    area(a),
    mask(mm),
    x(x),
    topo(bottom) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index i) const {
    if (not mask(i)) {
      const auto mcrd = Kokkos::subview(x, i, Kokkos::ALL);
      depth(i) = mass(i) / area(i);
      surface_height(i) = depth(i) + topo(mcrd);
    }
  }
};

/** Reduction functor for separate source and target points;

  This functor is called in the inner loop of a parallel_for( parallel_reduce ) pattern.
*/
template <typename Geo>
struct SWEVelocityReducerSeparatePts {
  static constexpr int ndim = Geo::ndim;
  using crd_view = typename Geo::crd_view_type;

  /** this value type collects the rhs terms for a swe velocity reduction

    values are packed as:

    value_type[0:ndim] = velocity components
    value_type[ndim:ndim*(1+ndim)] = ndim x ndim matrix for
      velocity gradient (row major storage)
  */
  using value_type = Kokkos::Tuple<Real, ndim*(1+ndim)>;
  Index i; /// index of tgt point in tgtx view
  crd_view tgtx; /// coordinates of tgt points that are all separated from
                 /// any source points
  crd_view srcx; /// coordinates of src points
  scalar_view_type rel_vort; /// relative vorticity of src points
  scalar_view_type divergence; /// divergence of src points
  scalar_view_type area; /// area of src points
  mask_view_type mask; /// mask to exclude divided src points
  Real eps; /// kernel regularization parameter

  KOKKOS_INLINE_FUNCTION
  SWEVelocityReducerSeparatePts(const Index i,
    const crd_view tx, const crd_view sx, const scalar_view_type zeta,
    const scalar_view_type sigma, const scalar_view_type a,
    const mask_view_type m, const Real eps) :
    i(i), tgtx(tx), srcx(sx), rel_vort(zeta), divergence(sigma),
    area(a), mask(m), eps(eps)
  {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index j, value_type& rhs) const {
    if (not mask(j)) {
      const auto tgt_crd = Kokkos::subview(tgtx, i, Kokkos::ALL);
      const auto src_crd = Kokkos::subview(srcx, j, Kokkos::ALL);
      const Real src_zeta = rel_vort(j);
      const Real src_sigma = divergence(j);
      const Real src_area = area(j);

      Real vel_zeta[ndim];
      SWEVelocity<Geo>::kzeta(vel_zeta, tgt_crd, src_crd, src_zeta, src_area, eps);

      Real vel_sigma[ndim];
      SWEVelocity<Geo>::ksigma(vel_sigma, tgt_crd, src_crd, src_sigma, src_area, eps);

      Real gkz[ndim*ndim];
      const Real vort_str = src_zeta*src_area;
      SWEVelocity<Geo>::grad_kzeta(gkz, tgt_crd, src_crd, eps);

      Real gks[ndim*ndim];
      const Real div_str = src_sigma*src_area;
      SWEVelocity<Geo>::grad_ksigma(gks, tgt_crd, src_crd, eps);

      for (Int i=0; i<ndim; ++i) {
        rhs[i] += vel_zeta[i] + vel_sigma[i];
      }
      for (Int i=0; i<ndim*ndim; ++i) {
        rhs[ndim + i] += gkz[i]*vort_str + gks[i]*div_str;
      }
    }
  }
};

/** Computes the velocity reduction at passive particles.

  This functor is called in the outer loop of a parallel_for(parallel_reduce) pattern.

  It computes velocity and the velocity gradient double dot product.
*/
template <typename Geo>
struct SWEVelocityPassive {
  static constexpr Int ndim = Geo::ndim;
  using crd_view = typename Geo::crd_view_type;
  using vec_view = typename Geo::vec_view_type;
  vec_view velocity;  /// output: velocity at tgt pts
  scalar_view_type double_dot; /// output: double dot product of velocity tensor at output pts
  crd_view tgtx; /// input: coordinates of target pts
  crd_view srcx; /// input: coordinates of source pts
  scalar_view_type rel_vort; /// input: vorticity of source pts
  scalar_view_type divergence; /// input: divergence of sources pts
  scalar_view_type area; /// input: area of src points
  mask_view_type mask; /// input: mask of src points
  Real eps; /// input: kernel regularization parameter
  Index nsrc; /// number of source points

  SWEVelocityPassive(vec_view u, scalar_view_type dd, crd_view tx, crd_view sx,
    scalar_view_type zeta, scalar_view_type sigma, scalar_view_type area,
    mask_view_type m, const Real eps, const Index n) :
      velocity(u),
      double_dot(dd),
      tgtx(tx),
      srcx(sx),
      rel_vort(zeta),
      divergence(sigma),
      area(area),
      mask(m),
      eps(eps),
      nsrc(n) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const member_type& mbr) const {
    const Index i = mbr.league_rank();  // index of tgt
    Kokkos::Tuple<Real, ndim*(1+ndim)> rhs; // collected results of reductions
    Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(mbr, nsrc),
      SWEVelocityReducerSeparatePts<Geo>(i, tgtx, srcx, rel_vort, divergence, area, mask, eps),
      rhs);
    auto mvel = Kokkos::subview(velocity, i, Kokkos::ALL);
    for (Int j=0; j<ndim; ++j) {
      mvel[j] = rhs[j];
    }
    double_dot[i] = 0;
    for (Int j=0; j<ndim; ++j) {
      for (Int k=0; k<ndim; ++k) {
        const Int jk_idx = j*ndim + k;
        const Int kj_idx = k*ndim + j;

        double_dot[i] += (j == k ? 1 : 2) * rhs[ndim + jk_idx] * rhs[ndim + kj_idx];
      }
    }
  }
};

template <typename Geo>
struct SWEVelocityReducerCollocatedPts {
  static constexpr int ndim = Geo::ndim;
  using crd_view = typename Geo::crd_view_type;
  /** this value type collects the rhs terms for a swe velocity reduction

    values are packed as:

    value_type[0:ndim] = velocity components
    value_type[ndim:ndim*(1+ndim)] = ndim x ndim matrix for
      velocity gradient (row major storage)
  */
  using value_type = Kokkos::Tuple<Real, ndim*(1+ndim)>;
  Index i; /// index of tgt point in tgtx view
  crd_view srcx; /// coordinates of src/tgt points
  scalar_view_type rel_vort; /// relative vorticity of src points
  scalar_view_type divergence; /// divergence of src points
  scalar_view_type area; /// area of src points
  mask_view_type mask; /// mask to exclude divided src points
  Real eps; /// kernel regularization parameter

  KOKKOS_INLINE_FUNCTION
  SWEVelocityReducerCollocatedPts(const Index i,
    const crd_view sx, const scalar_view_type zeta,
    const scalar_view_type sigma, const scalar_view_type a,
    const mask_view_type m, const Real eps) :
    i(i), srcx(sx), rel_vort(zeta), divergence(sigma),
    area(a), mask(m), eps(eps)
  {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index j, value_type& rhs) const {
    if ( (not mask(j) and (i != j)) ) {
      const auto tgt_crd = Kokkos::subview(srcx, i, Kokkos::ALL);
      const auto src_crd = Kokkos::subview(srcx, j, Kokkos::ALL);
      const Real src_zeta = rel_vort(j);
      const Real src_sigma = divergence(j);
      const Real src_area = area(j);

      Real vel_zeta[ndim];
      SWEVelocity<Geo>::kzeta(vel_zeta, tgt_crd, src_crd, src_zeta, src_area, eps);

      Real vel_sigma[ndim];
      SWEVelocity<Geo>::ksigma(vel_sigma, tgt_crd, src_crd, src_sigma, src_area, eps);

      Real gkz[ndim*ndim];
      const Real vort_str = src_zeta*src_area;
      SWEVelocity<Geo>::grad_kzeta(gkz, tgt_crd, src_crd, eps);

      Real gks[ndim*ndim];
      const Real div_str = src_sigma*src_area;
      SWEVelocity<Geo>::grad_ksigma(gks, tgt_crd, src_crd, eps);

      for (Int i=0; i<ndim; ++i) {
        rhs[i] += vel_zeta[i] + vel_sigma[i];
      }
      for (Int i=0; i<ndim*ndim; ++i) {
        rhs[ndim + i] += gkz[i]*vort_str + gks[i]*div_str;
      }
    }
  }
};

template <typename Geo>
struct SWEVelocityActive {
  static constexpr Int ndim = Geo::ndim;
  using crd_view = typename Geo::crd_view_type;
  using vec_view = typename Geo::vec_view_type;
  vec_view velocity;  /// output: velocity at tgt pts
  scalar_view_type double_dot; /// output: double dot product of velocity tensor at output pts
  crd_view srcx; /// input: coordinates of source pts
  scalar_view_type rel_vort; /// input: vorticity of source pts
  scalar_view_type divergence; /// input: divergence of sources pts
  scalar_view_type area; /// input: area of src points
  mask_view_type mask; /// input: mask of src points
  Real eps; /// input: kernel regularization parameter
  Index nsrc; /// number of source points

  SWEVelocityActive(vec_view u, scalar_view_type dd, crd_view sx,
    scalar_view_type zeta, scalar_view_type sigma, scalar_view_type area,
    mask_view_type m, const Real eps, const Index n) :
      velocity(u),
      double_dot(dd),
      srcx(sx),
      rel_vort(zeta),
      divergence(sigma),
      area(area),
      mask(m),
      eps(eps),
      nsrc(n) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const member_type& mbr) const {
    const Index i = mbr.league_rank();  // index of tgt
    Kokkos::Tuple<Real, ndim*(1+ndim)> rhs; // collected results of reductions

    // compute rhs velocity reductions
    Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(mbr, nsrc),
      SWEVelocityReducerCollocatedPts<Geo>(i, srcx, rel_vort, divergence, area, mask, eps),
      rhs);

    // unpack results
    auto mvel = Kokkos::subview(velocity, i, Kokkos::ALL);
    for (Int j=0; j<ndim; ++j) {
      mvel[j] = rhs[j];
    }
    double_dot[i] = 0;
    for (Int j=0; j<ndim; ++j) {
      for (Int k=0; k<ndim; ++k) {
        const Int jk_idx = j*ndim + k;
        const Int kj_idx = k*ndim + j;

        double_dot[i] += (j == k ? 1 : 2) * rhs[ndim + jk_idx] * rhs[ndim + kj_idx];
      }
    }
  }
};




} // namespace Lpm

#endif
