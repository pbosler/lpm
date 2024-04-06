#ifndef LPM_CORIOLIS_HPP
#define LPM_CORIOLIS_HPP

#include "LpmConfig.h"
#include "lpm_geometry.hpp"
#include "lpm_constants.hpp"

namespace Lpm {

// *  Coriolis parameter for the beta plane
//
//   @f$ f = f_0 + \beta y @f$
//
//   For a reference latitude phi0, the parameters are
//   @f$ f_0 = 2\Omega\sin\phi_0,\qquad \beta = 2\Omega\cos\phi_0 @f$.
//
//   @param [in] f0
//   @param [in] beta
//   @return f = f0 + beta * y
// */
template <typename Geo>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<std::is_same<Geo,PlaneGeometry>::value, Real>::type
coriolis_f(const Real f0, const Real beta, const Real y) {
  return f0 + beta * y;
}

/**  Time derivative of the Coriolis parameter of a parcel for the beta plane

  @f$ \frac{Df}{Dt} = \beta \frac{Dy}{Dt} @f$

  For a reference latitude phi0, the parameters are
  @f$ f_0 = 2\Omega\sin\phi_0,\qquad \beta = 2\Omega\cos\phi_0 @f$.

  @param [in] beta
  @param [in] v meriodional component of velocity
  @return Df/Dt = beta * v
*/
template <typename Geo>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<std::is_same<Geo,PlaneGeometry>::value, Real>::type
coriolis_dfdt(const Real beta, const Real v) {
  return beta * v;
}

// * Coriolis parameter for the rotating sphere.
//
//   @f$ f = 2\Omega z @f$
//   where @f$\Omega@f$ is the constant angular velocity of the rotation about
//   the positive z-axis.
//
//   @param [in] Omega rotational velocity
//   @param [in] z z-coordinate of a Lagrangian parcel
//   @return f = 2 * Omega * z
// */
template <typename Geo>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<std::is_same<Geo,SphereGeometry>::value, Real>::type
coriolis_f(const Real Omega, const Real z) {
  return 2 * Omega * z;
}

/** Time derivative of the Coriolis parameter for a Lagrangian parcel on
  a rotating sphere.

  @f$ \frac{Df}{Dt} = 2\Omega\frac{Dz}{Dt} @f$

  @param [in] Omega rotational velocity
  @param [in] w z-component of velocity
  @return df/dt = 2 * Omega * w
*/
template <typename Geo>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<std::is_same<Geo,SphereGeometry>::value, Real>::type
coriolis_dfdt(const Real Omega, const Real w) {
  return 2 * Omega * w;
}

/** @brief Coriolis parameter definition and evaluation
  for problems on the beta plane.
*/
struct CoriolisBetaPlane {
  Real f0;
  Real beta;
  static constexpr Real Omega = 2*constants::PI;

  KOKKOS_INLINE_FUNCTION
  CoriolisBetaPlane() : f0(0), beta(0) {}

  KOKKOS_INLINE_FUNCTION
  explicit CoriolisBetaPlane(const Real phi0) :
    f0(2*Omega*sin(phi0)),
    beta(2*Omega*cos(phi0)) {}

  KOKKOS_INLINE_FUNCTION
  CoriolisBetaPlane(const Real f0, const Real beta) :
    f0(f0), beta(beta) {}

  KOKKOS_INLINE_FUNCTION
  CoriolisBetaPlane(const CoriolisBetaPlane& other) = default;

  KOKKOS_INLINE_FUNCTION
  Real f(const Real y) const {return f0 + beta*y;}

  template <typename PtType>
  KOKKOS_INLINE_FUNCTION
  Real f(const PtType& xy) const {return f0 + beta*xy[1];}

  KOKKOS_INLINE_FUNCTION
  Real dfdt(const Real v) const {return beta * v;}

  template <typename PtType>
  KOKKOS_INLINE_FUNCTION
  Real dfdt(const PtType& uv) {return beta * uv[1];}
};

struct CoriolisSphere {
  Real Omega;

  KOKKOS_INLINE_FUNCTION
  explicit CoriolisSphere(const Real Omg=2*constants::PI) :
    Omega(Omg) {}

  KOKKOS_INLINE_FUNCTION
  CoriolisSphere(const CoriolisSphere& other) = default;

  KOKKOS_INLINE_FUNCTION
  Real f (const Real z) const {return 2*Omega*z;}

  template <typename PtType>
  KOKKOS_INLINE_FUNCTION
  Real f (const PtType& xyz) const {return 2*Omega*xyz[2];}

  KOKKOS_INLINE_FUNCTION
  Real dfdt(const Real& w) const {return 2*Omega*w;}

  template <typename PtType>
  KOKKOS_INLINE_FUNCTION
  Real dfdt (const PtType& uvw) const {return 2*Omega*uvw[2];}
};




} // namespace Lpm

#endif
