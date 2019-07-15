#ifndef LPM_SPHERE_BIOT_SAVART_HPP
#define LPM_SPHERE_BIOT_SAVART_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmPolyMesh2d.hpp"
#include "LpmKokkosUtil.hpp"
#include "LpmGeometry.hpp"
#include "Kokkos_Core.hpp"
#include "LpmVtkIO.hpp"
#include <cmath>
#include <iomanip>

namespace Lpm {

  typedef typename SphereGeometry::crd_view_type crd_view;
  typedef typename SphereGeometry::crd_view_type vec_view_type;
  typedef typename ko::TeamPolicy<>::member_type member_type;

  template <typename VT, typename CVT> KOKKOS_INLINE_FUNCTION
  void biotSavartKernel(VT& result, const CVT& tgt_x, const CVT& src_xx, const Real& src_vort, const Real& src_area,
    const Real eps) {
    const Real denom = -4*PI*(1.0 - SphereGeometry::dot(tgt_x, src_xx) + square(eps));
    const Real str = src_vort*src_area/denom;
    const ko::Tuple<Real,3> cp  = SphereGeometry::cross(result, tgt_x, src_xx);
    for (int j=0; j<3; ++j)
      result[j] = cp[j]*str;
  }

  KOKKOS_INLINE_FUNCTION
  Real legendre54(const Real& z) {
    return z*square((square(z)-1));
  }
  
  template <typename VT> KOKKOS_INLINE_FUNCTION
  Real SphHarm54(const VT& x) {
    const Real lam = SphereGeometry::longitude(x);
    return 30*std::cos(4*lam)*legendre54(x[2]);
  }

  template <typename VT, typename CVT> KOKKOS_INLINE_FUNCTION
  void rh54Velocity(VT& u, const CVT& x) {
    const Real theta = SphereGeometry::latitude(x);
    const Real lambda = SphereGeometry::longitude(x);
    const Real usph = 0.5*std::cos(4*lambda)*cube(std::cos(theta))*(-3 + 5*std::cos(2*theta));
    const Real vsph = 4*cube(std::cos(theta))*std::sin(4*lambda)*std::sin(theta)
    u[0] = -usph*std::sin(lambda) - vsph*std::sin(theta)*std::cos(lambda);
    u[1] =  usph*std::cos(lambda) - vsph*std::sin(theta)*std::sin(lambda);
    u[2] =  vpsh*std::cos(theta);
  }

  struct Init {
    scalar_view_type zeta;
    vec_view_type exactu;
    crd_view x;

    Init(scalar_view_type zz, vec_view_type u, crd_view xx) : zeta(zz), exactu(u), x(xx) {}

    KOKKOS_INLINE_FUNCTION
    void operator() (const Index i) const {
      auto myx = ko::subview(x,i,ko::ALL());
      zeta(i) = SphHarm54(myx);
      auto myu = ko::subview(exactu,i,ko::ALL());
      rh54Velocity(myu, myx);
    }
    
  };
  
}
#endif
