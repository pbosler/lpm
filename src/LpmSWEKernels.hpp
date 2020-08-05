#ifndef LPM_SWE_KERNELS_HPP
#define LPM_SWE_KERNELS_HPP
#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmKokkosUtil.hpp"
#include "Kokkos_Core.hpp"
#include <cmath>

namespace Lpm {

typedef typename SphereGeometry::crd_view_type crd_view;
typedef typename SphereGeometry::crd_view_type vec_view;
typedef typename ko::TeamPolicy<>::member_type member_type;

template <typename VecType> KOKKOS_INLINE_FUNCTION
void sweVelocityKernel(ko::Tuple<Real,3>& u, const VecType& tgt, const VecType& src,
  const Real& src_vort, const Real& src_div, const Real& src_area) {
  const Real area_over_denom = src_area/(4*PI*(1-SphereGeometry::dot(tgt,src)));
  ko::Tuple<Real,3> kzeta = SphereGeometry::cross(tgt,src);
  ko::Tuple<Real,3> ksigma;
  ksigma[0] =  src(0)*(1-square(tgt(0))) - src(1)*tgt(0)*tgt(1) - src(2)*tgt(0)*tgt(2);
  ksigma[1] = -src(0)*tgt(0)*tgt(1) + src(1)*(1-square(tgt(1))) - src(2)*tgt(1)*tgt(2);
  ksigma[2] = -src(0)*tgt(0)*tgt(2) - src(1)*tgt(1)*tgt(2) + src(2)*(1-square(tgt(2)));
  for (Short j=0; j<3; ++j) {
    u[j] = (-kzeta[j]*src_vort + ksigma[j]*src_div)*area_over_denom;
  }
}

template <typename MT,typename CT> KOKKOS_INLINE_FUNCTION
void gradKernelMats(MT& gkzeta, MT& gksigma, const CT& x, const CT& y) {
  const Real oodenom = 1/(4*PI*square(1-SphereGeometry::dot(x,y)));

  gkzeta(0,0) = -(x(0) - y(0))*(x(2)*y(1) - x(1)*y(2));
  gkzeta(0,1) = square(x(1))*y(2) - x(2)*x(1)*y(1) + y(2)*(x(0)*y(0) - 1) +
    x(2)*(square(y(1)) + square(y(2)));
  gkzeta(0,2) = -x(1)*square(y(1)) - y(1)*(x(0)*y(0) + square(x(2)) - 1) +
    x(1)*y(2)*(x(2)-y(2));

  gkzeta(1,0) = -square(x(0))*y(2) + x(2)*x(0)*y(0) + y(2)*(1 - x(1)*y(1)) -
    x(2)*(square(y(0)) + square(y(2)));
  gkzeta(1,1) = (x(1) - y(1))*(x(2)*y(0) - x(0)*y(2));
  gkzeta(1,2) = x(0)*square(y(0)) + y(0)*(x(1)*y(1) + square(x(2)) - 1) +
    x(0)*y(2)*(y(2) - x(2));

  gkzeta(2,0) = square(x(0))*y(1) - x(1)*x(0)*y(0) + x(1)*(square(y(0)) + square(y(1))) +
    y(1)*(x(2)*y(2) - 1);
  gkzeta(2,1) = -x(0)*square(y(0)) - y(0)*(x(2)*y(2) + square(x(1)) - 1) +
    x(0)*y(1)*(x(1) - y(1));
  gkzeta(2,2) = -(x(1)*y(0) - x(0)*y(1))*(x(2) - y(2));

  gksigma(0,0) = -std::pow(x(0),4)*square(y(0)) - 2*cube(x(0))*y(0) *
    (x(1)*y(1) + x(2)*y(2) - 1) - square(x(0))*(square(x(1)*y(1)) +
    2*x(1)*y(1)*(x(2)*y(2) - 1) + x(2)*y(2)*(x(2)*y(2) - 2)) +
    x(0)*y(0)*(x(1)*y(1) + x(2)*y(2) - 2) + square(x(1)*y(1)) +
    square(x(2)*y(2)) - x(1)*y(1) - x(2)*y(2) + 2*x(1)*x(2)*y(1)*y(2) +
    square(y(2));
  gksigma(0,1) = -x(1)*cube(x(0))*square(y(0)) - 2*x(1)*square(x(0))*y(0)*
    (x(1)*y(1) + x(2)*y(2)-1) - x(0)*(cube(x(1))*square(y(1)) + 2*square(x(1))*y(1)*
    (x(2)*y(2) - 1) + x(1)*(x(2)*y(2)*(x(2)*y(2)-2)+square(y(0))) + y(1)) +
    y(0)*(-(square(x(1))-1)*y(1) - x(1)*x(2)*y(2));
  gksigma(0,2) = -x(2)*cube(x(0))*square(y(0)) - 2*x(2)*square(x(0))*y(0)*
    (x(1)*y(1) + x(2)*y(2)-1) - x(0)*(cube(x(2))*square(y(2)) + 2*square(x(2))*y(2)*
    (x(1)*y(1) - 1) + x(2)*(x(1)*y(1)*(x(1)*y(1)-2) + square(y(0))) + y(2)) +
    y(0)*(-x(1)*x(2)*y(1) - (square(x(2))-1)*y(2));

  gksigma(1,0) = -x(0)*cube(x(1))*square(y(1)) - 2*x(0)*square(x(1))*y(1)*
    (x(0)*y(0) + x(2)*y(2) - 1) - x(1)*(cube(x(0))*square(y(0)) + x(0)*
    (x(2)*y(2)*(x(2)*y(2)-2) + square(y(1))) + y(0)*(2*square(x(0))*(x(2)*y(2)-1)+1)) +
    y(1)*(-(square(x(0))-1)*y(0) - x(0)*x(2)*y(2));
  gksigma(1,1) = -std::pow(x(1),4)*square(y(1)) - 2*cube(x(1))*y(1)*(x(2)*y(2)-1) +
    x(2)*square(x(1))*y(2)*(2-x(2)*y(2)) + x(1)*y(1)*(x(2)*y(2)-2) - square(x(0))*
    (square(x(1))-1)*square(y(0)) + square(x(0)*y(0)) - x(2)*y(2) + x(0)*y(0)*
    (-2*cube(x(1))*y(1) + square(x(1))*(2-2*x(2)*y(2)) + x(1)*y(1) + 2*x(2)*y(2) - 1) +
    square(y(1));
  gksigma(1,2) = -x(2)*cube(x(1))*square(y(1))-2*x(2)*square(x(1))*y(1)*(x(2)*y(2)-1) -
    square(x(0))*x(2)*x(1)*square(y(0)) - x(1)*(cube(x(2))*square(y(2))-2*square(x(2))*
    y(2) + x(2)*square(y(1))+y(2)) - (square(x(2))-1)*y(1)*y(2) - x(0)*x(2)*y(0)*
    (2*square(x(1))*y(1) + 2*x(1)*(x(2)*y(2)-1) + y(1));

  gksigma(2,0) = -x(0)*cube(x(2))*square(y(2)) - 2*x(0)*square(x(2))*y(2)*
    (x(0)*y(0)+x(1)*y(1)-1) - x(2)*(cube(x(0))*square(y(0)) + x(0)*
    (square(x(1)*y(1)) - 2*x(1)*y(1) +square(y(2))) + y(0)*(2*square(x(0))*
    (x(1)*y(1)-1)+1)) + y(2)*(-(square(x(0))-1)*y(0)-x(0)*x(1)*y(1));
  gksigma(2,1) = -x(1)*cube(x(2))*square(y(2)) - 2*x(1)*square(x(2))*y(2)*(x(1)*y(1)-1) -
    square(x(0))*x(1)*x(2)*square(y(0)) - x(2)*(cube(x(1))*square(y(1)) - 2*square(x(1))*
    y(1) + x(1)*square(y(2)) + y(1)) - (square(x(1))-1)*y(1)*y(2) - x(0)*x(1)*y(0)*
    (2*square(x(2))*y(2) + 2*x(2)*(x(1)*y(1)-1) + y(2));
  gksigma(2,2) = -square(x(0))*(square(x(2))-1)*square(y(0)) + x(0)*y(0)*
    (-2*cube(x(2))*y(2) + square(x(2))*(2-2*x(1)*y(1)) + x(2)*y(2) + 2*x(1)*y(1) - 1) -
    square(x(1))*(square(x(2))-1)*square(y(1)) - x(1)*(2*square(x(2))-1)*y(1)*(x(2)*y(2)-1) +
    y(2)*(-std::pow(x(2),4)*y(2) + 2*cube(x(2)) - 2*x(2) + y(2));

  for (Short ii=0; ii<3; ++ii) {
    for (Short jj=0; jj<3; ++jj) {
      gkzeta(ii,jj) *= oodenom;
      gksigma(ii,jj) *= oodenom;
    }
  }
}

template <typename MatrixType, typename TgtVecType, typename SrcVecType>
struct SweKernelGradFunctor {
  MatrixType gkzeta; /// 3 x 3
  MatrixType gksigma; //// 3 x 3
  TgtVecType tgtx; /// tgt coordinates
  SrcVecType srcy; /// src coordinates
  mask_view_type src_mask;
  Index i; /// index of target

  struct DistinctTag {};
  struct CollocatedTag {};

  KOKKOS_INLINE_FUNCTION
  void operator() (const DistinctTag&, const Index& j) const {
    const auto x = ko::subview(tgtx, i, ko::ALL());
    const auto y = ko::subview(srcx, j, ko::ALL());

    if (src_mask(j)) {
      for (Short ii=0; ii<3; ++ii) {
        for (Short jj=0; jj<3;; ++jj) {
          gkzeta(ii,jj) = 0.0;
          gksigma(ii,jj) = 0.0;
        }
      }
    }
    else {
      gradKernelMats(gkzeta, gksigma, x, y);
    }

  KOKKOS_INLINE_FUNCTION
  void operator() (const CollocatedTag&, const Index& j) const {
    const auto x = ko::subview(tgtx, i, ko::ALL());
    const auto y = ko::subview(srcx, j, ko::ALL());

    if (src_mask(j) || i==j) {
      for (Short ii=0; ii<3; ++ii) {
        for (Short jj=0; jj<3;; ++jj) {
          gkzeta(ii,jj) = 0.0;
          gksigma(ii,jj) = 0.0;
        }
      }
    }
    else {
      gradKernelMats(gkzeta, gksigma, x, y);
    }
};


}
#endif
