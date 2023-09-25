#ifndef LPM_SWE_GALLERY_HPP
#define LPM_SWE_GALLERY_HPP

#include "LpmConfig.h"
#include "util/lpm_floating_point.hpp"

namespace Lpm {

struct PlanarGravityWaveFreeBoundaries {
  typedef PlaneGeometry geo;
  Real mtn_height;
  Real mtn_width_param;
  Real mtn_ctr_x;
  Real mtn_ctr_y;
  Real sfc_height;
  Real sfc_ptb_max;
  Real sfc_ptb_width_bx;
  Real sfc_ptb_width_by;
  Real sfc_ptb_ctr_x;
  Real sfc_ptb_ctr_y;
  Real f0;
  Real beta;

  KOKKOS_INLINE_FUNCTION
  PlanarGravityWaveFreeBoundaries() :
    mtn_height(0.8),
    mtn_width_param(5),
    mtn_ctr_x(0),
    mtn_ctr_y(0),
    sfc_height(1),
    sfc_ptb_max(0.1),
    sfc_ptb_width_bx(20),
    sfc_ptb_width_by(5),
    sfc_ptb_ctr_x(-1.125),
    sfc_ptb_ctr_y(0),
    f0(0),
    beta(0) {}

  inline std::string name() const {return "PlanarGravityWaveFreeBoundaries";}

  template <typename PtType>
  KOKKOS_INLINE_FUNCTION
  Real bottom_height(const PtType& xy) const {
    return mtn_height*exp(-mtn_width_param*(square(xy[0]-mtn_ctr_x) + square(xy[1] - mtn_ctr_y)));
  }

  template <typename PtType>
  KOKKOS_INLINE_FUNCTION
  Real sfc0(const PtType& xy) const { return
    sfc_height + sfc_ptb_max*exp(-(sfc_ptb_width_bx*square(xy(0) - sfc_ptb_ctr_x) + sfc_ptb_width_by*square(xy(1) - sfc_ptb_ctr_y)));
  }

  template <typename PtType>
  KOKKOS_INLINE_FUNCTION
  Real zeta0(const PtType& xy) const { return 0;}

  template <typename PtType, typename VecType>
  KOKKOS_INLINE_FUNCTION
  void u0(VecType& uv, const PtType& xy) const {
    uv[0] = 0;
    uv[1] = 0;
  }

  template <typename PtType>
  KOKKOS_INLINE_FUNCTION
  Real sigma0(const PtType& xy) const { return 0;}
};


struct NitscheStricklandVortex {
  typedef PlaneGeometry geo;
  Real vortex_width_b;
  Real sfc_height;

  KOKKOS_INLINE_FUNCTION
  NitscheStricklandVortex(const Real b = 0.5, const Real sfc = 1) :
    vortex_width_b(0.5),
    sfc_height(sfc) {}

  template <typename PtType>
  KOKKOS_INLINE_FUNCTION
  Real zeta0(const PtType& xy) const {
    const Real rsq = square(xy[0]) + square(xy[1]);
    const Real r = sqrt(rsq);
    return (3 * safe_divide(r) - 2 * vortex_width_b * r) * rsq * std::exp(-vortex_width_b * rsq);
  }

  inline std::string name() const { return "Nitsche&Strickland"; }

  template <typename VecType, typename PtType>
  KOKKOS_INLINE_FUNCTION
  void u0(VecType& uv, const PtType& xy) const {
    const Real rsq = square(xy[0]) + square(xy[1]);
    const Real utheta = rsq * std::exp(-vortex_width_b * rsq);
    const Real theta = std::atan2(xy[1], xy[0]);
    uv[0] = -utheta * std::sin(theta);
    uv[1] = utheta * std::cos(theta);
  }

  template <typename PtType>
  KOKKOS_INLINE_FUNCTION
  Real sigma0(const PtType& xy) const {return 0;}

  template <typename PtType>
  KOKKOS_INLINE_FUNCTION
  Real sfc0(const PtType& xy) const {return sfc_height; }

  template <typename PtType>
  KOKKOS_INLINE_FUNCTION
  Real h0(const PtType& xy) const {return
    sfc0(xy);
  }

  template <typename PtType>
  KOKKOS_INLINE_FUNCTION
  Real bottom_height(const PtType& xy) const {return 0;}
};



} // namespace Lpm

#endif
