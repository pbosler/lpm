#ifndef LPM_SWE_GALLERY_HPP
#define LPM_SWE_GALLERY_HPP
#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmUtilities.hpp"
#include "Kokkos_Core.hpp"
#include <memory>
#include <cmath>

namespace Lpm {

struct PlanarGaussianMtnTopo {
  static constexpr Real height = 0.25;
  static constexpr Real ctr_x = 0.75;
  static constexpr Real ctr_y = 0.0;
  static constexpr Real b = 5.0;

  static Real eval(const Real& x, const Real& y) {
    return height*std::exp(-b*(square(x-ctr_x) + square(y-ctr_y)));
  }
};

struct Thacker81BasinTopo {
  static constexpr Real max_depth = 0.5;
  static constexpr Real x_radius = 0.75;
  static constexpr Real y_radius = 0.75;
  static constexpr Real xr2 = x_radius*x_radius;
  static constexpr Real yr2 = y_radius*y_radius;

  KOKKOS_INLINE_FUNCTION
  static Real eval(const Real& x, const Real& y) {
    return -max_depth*(1-square(x)/xr2 - square(y)/yr2);
  }

  template <typename VT> KOKKOS_INLINE_FUNCTION
  static Real eval(const VT& xy) {
    return -max_depth * (1 - square(xy(0))/xr2 - square(xy(1))/yr2);
  }
};

struct FlatTopo {
  template <typename VT> KOKKOS_INLINE_FUNCTION
  static Real eval(const VT& xy) {return 0;}
};

// struct SfcInitialCondition {
//   typedef std::shared_ptr<SfcInitialCondition> ptr;
//
//   virtual ~SfcInitialCondition() {}
//
//   virtual Real eval_sfc(const Real& x, const Real& y, const Real& z) const = 0;
//   virtual Real eval_h(const Real& x, const Real& y, const Real& z) const = 0;
// };

// struct Thacker81FlatOscillation : public SfcInitialCondition {
//   static constexpr Real L = Thacker81BasinTopo::x_radius;
//   static constexpr Real D0 = Thacker81BasinTopo::max_depth;
//   static constexpr Real eta = 0.5*L;
//
//   Real eval_sfc(const Real& x, const Real& y, const Real& z) const;
//
//   Real eval_h(const Real& x, const Real& y, const Real& z) const;
//
// //   Real exact_sfc(const Real& x, const Real& y, const Real& t) const;
// //   Real exact_h(const Real& x, const Real& y, const Real& t) const;
// //   void exact_velocity(Real& u, Real& v, const Real& x, const Real& y, const Real& t) const;
// };

struct Thacker81CurvedOscillation {
  static constexpr Real L = 0.75;
  static constexpr Real D0 = 0.5;
  static constexpr Real eta = 0.125*L;
  static constexpr Real g = 1.0;
  static constexpr Real f = 0.125*(g*D0/(L*L));
  static constexpr Real beta = 0;
  static constexpr Real OMEGA = 0;
  static constexpr Real A = ((D0+eta)*(D0+eta) - D0*D0)/((D0+eta)*(D0+eta)+D0*D0);

  template <typename CV> KOKKOS_INLINE_FUNCTION
  static Real bottom_height(const CV& xy) {
    return -D0 * (1 - (square(xy(0)) + square(xy(1)))/(L*L));
  }

  template <typename CV> KOKKOS_INLINE_FUNCTION
  static Real zeta0(const CV& xy) {
    return 1/(1-A)*(f*std::sqrt(1-A*A)+A-1);
  }

  template <typename CV> KOKKOS_INLINE_FUNCTION
  static Real sigma0(const CV& xy) {
    return 0.0;
  }

  template <typename CV> KOKKOS_INLINE_FUNCTION
  static Real sfc0(const CV& xy) {
    return D0*(std::sqrt(1-A*A)/(1-A) - 1 - (square(xy(0)) + square(xy(1)))/L*L*
      ((1-A*A)/square(1-A)-1));
  }

  template <typename CV> KOKKOS_INLINE_FUNCTION
  static Real h0(const CV& xy) {
    return sfc0(xy) - bottom_height(xy);
  }


  KOKKOS_INLINE_FUNCTION
  static Real omega() {return std::sqrt(8*g*D0/L*L + f*f);}

  template <typename V, typename CV> KOKKOS_INLINE_FUNCTION
  static void exact_velocity(V& uv, const CV& xy, const Real& t=0) {
    const Real omg = omega();
    const Real cosomgt = std::cos(omg*t);
    const Real sinomgt = std::sin(omg*t);
    const Real a0 = 1/(1 - A*cosomgt);
    const Real a1 = sqrt(1-A*A) + A*cosomgt - 1;
    uv(0) = a0*(0.5*omg*xy(0)*A*sinomgt - 0.5*f*xy(1)*a1);
    uv(1) = a0*(0.5*omg*xy(1)*A*sinomgt + 0.5*f*xy(0)*a1);
  }

  template <typename CV> KOKKOS_INLINE_FUNCTION
  static Real exact_zeta(const CV& xy, const Real& t) {
    const Real omg = omega();
    const Real cosomgt = std::cos(omg*t);
    return f*(std::sqrt(1-A*A) + A*cosomgt-1)/(1-A*cosomgt);
  }

  template <typename CV> KOKKOS_INLINE_FUNCTION
  static Real exact_sigma(const CV& xy, const Real& t) {
    const Real omg = omega();
    return omg*A*std::sin(omg*t)/(1-A*std::cos(omg*t));
  }

  template <typename CV> KOKKOS_INLINE_FUNCTION
  static Real exact_sfc(const CV& xy, const Real& t) {
    const Real omg = omega();
    const Real cosomgt = std::cos(omg*t);
    return D0*(std::sqrt(1-A*A)/(1-A*cosomgt) -1 - (square(xy(0)) + square(xy(1)))/L*L*
      ((1-A*A)/square(1-A*cosomgt)-1));
  }

  template <typename CV> KOKKOS_INLINE_FUNCTION
  static Real exact_h(const CV& xy, const Real& t) {
    return exact_sfc(xy,t) - bottom_height(xy);
  }
};

}
#endif
