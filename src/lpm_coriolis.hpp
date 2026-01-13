#ifndef LPM_CORIOLIS_HPP
#define LPM_CORIOLIS_HPP

#include "LpmConfig.h"
#include "lpm_constants.hpp"
#include "lpm_geometry.hpp"

namespace Lpm {

/**  Coriolis parameter for the beta plane

   @f$ f = f_0 + \beta y @f$

   @deprecated in favor of type implementations

   For a reference latitude phi0, the parameters are
   @f$ f_0 = 2\Omega\sin\phi_0,\qquad \beta = 2\Omega\cos\phi_0 @f$.

   @param [in] f0
   @param [in] beta
   @return f = f0 + beta * y
 */
template <typename Geo>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<std::is_same<Geo, PlaneGeometry>::value, Real>::type
    coriolis_f(const Real f0, const Real beta, const Real y) {
  return f0 + beta * y;
}

/**  Time derivative of the Coriolis parameter of a parcel for the beta plane

  @f$ \frac{Df}{Dt} = \beta \frac{Dy}{Dt} @f$

  @deprecated in favor of type implementations

  For a reference latitude phi0, the parameters are
  @f$ f_0 = 2\Omega\sin\phi_0,\qquad \beta = 2\Omega\cos\phi_0 @f$.

  @param [in] beta
  @param [in] v meriodional component of velocity
  @return Df/Dt = beta * v
*/
template <typename Geo>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<std::is_same<Geo, PlaneGeometry>::value, Real>::type
    coriolis_dfdt(const Real beta, const Real v) {
  return beta * v;
}

/** Coriolis parameter for the rotating sphere.

   @f$ f = 2\Omega z @f$
   where @f$\Omega@f$ is the constant angular velocity of the rotation about
   the positive z-axis.

   @deprecated in favor of type implementations

   @param [in] Omega rotational velocity
   @param [in] z z-coordinate of a Lagrangian parcel
   @return f = 2 * Omega * z
*/
template <typename Geo>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<std::is_same<Geo, SphereGeometry>::value,
                            Real>::type
    coriolis_f(const Real Omega, const Real z) {
  return 2 * Omega * z;
}

/** Time derivative of the Coriolis parameter for a Lagrangian parcel on
  a rotating sphere.

  @f$ \frac{Df}{Dt} = 2\Omega\frac{Dz}{Dt} @f$

  @deprecated in favor of type implementations

  @param [in] Omega rotational velocity
  @param [in] w z-component of velocity
  @return df/dt = 2 * Omega * w
*/
template <typename Geo>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<std::is_same<Geo, SphereGeometry>::value,
                            Real>::type
    coriolis_dfdt(const Real Omega, const Real w) {
  return 2 * Omega * w;
}

/** @brief Coriolis parameter definition and evaluation
  for problems on the beta plane.

  @f$ f(x) = f_0 + \beta * x_2 @f$

  @f$ df/df = \beta * d x_2 / dt @f$

*/
struct CoriolisBetaPlane {
  /// f-plane parameter, or central latitude parameter for beta plane
  Real f0;
  /// beta parameter
  Real beta;
  /// reference rotation to relate f0 and beta to a base latitude
  static constexpr Real Omega = 2 * constants::PI;

  /// default constructor disables Coriolis
  KOKKOS_INLINE_FUNCTION
  CoriolisBetaPlane() : f0(0), beta(0) {}

  /** constructor, sets parameters based on a central latitude

    @param [in] phi0 base latitude (radians)
  */
  KOKKOS_INLINE_FUNCTION
  explicit CoriolisBetaPlane(const Real phi0)
      : f0(2 * Omega * sin(phi0)), beta(2 * Omega * cos(phi0)) {}

  /** constructor with explicit values for f0 and beta.
   */
  KOKKOS_INLINE_FUNCTION
  CoriolisBetaPlane(const Real f0, const Real beta) : f0(f0), beta(beta) {}

  /// helpful for GPU (makes copy constructor available on device).
  KOKKOS_INLINE_FUNCTION
  CoriolisBetaPlane(const CoriolisBetaPlane& other) = default;

  /** Evaluate Coriolis parameter at a given position

    @param [in] xy position
  */
  template <typename PtType>
  KOKKOS_INLINE_FUNCTION Real f(const PtType& xy) const {
    return f0 + beta * xy[1];
  }

  /** Evaluate Coriolis parameter time derivative for a  given velocity

    @param [in] uv velocity
  */
  template <typename PtType>
  KOKKOS_INLINE_FUNCTION Real dfdt(const PtType& uv) const {
    return beta * uv[1];
  }

  template <typename XType, typename UType>
  KOKKOS_INLINE_FUNCTION Real grad_f_cross_u(const XType& x,
                                             const UType& u) const {
    return -beta * u[1];
  }
};

/**  Coriolis evaluation for a sphere with constant background
  rotation about the positive z-axis.
*/
struct CoriolisSphere {
  /// Angular velocity of sphere
  Real Omega;

  /** Default constructor initializes period of rotation to 2*pi or
    user-specified value.

    @param [in] Omg constant angular velocity
  */
  KOKKOS_INLINE_FUNCTION
  explicit CoriolisSphere(const Real Omg = 2 * constants::PI) : Omega(Omg) {}

  /// helpful for GPU (makes copy constructor available on device).
  KOKKOS_INLINE_FUNCTION
  CoriolisSphere(const CoriolisSphere& other) = default;

  /** Evaluate Coriolis parameter at a given position

    @param [in] xyz position
  */
  template <typename PtType>
  KOKKOS_INLINE_FUNCTION Real f(const PtType& xyz) const {
    return 2 * Omega * xyz[2];
  }

  /** Evaluate Coriolis parameter material derivative
   */
  template <typename UType>
  KOKKOS_INLINE_FUNCTION Real dfdt(const UType& u) const {
    return 2 * Omega * u[2];
  }

  /** Evaluate Coriolis parameter time derivative for a given velocity

    @param [in] uvw velocity
  */
  template <typename XType, typename UType>
  KOKKOS_INLINE_FUNCTION Real grad_f_cross_u(const XType& x,
                                             const UType& u) const {
    return -2 * Omega * (-u[0] * x[1] + u[1] * x[0]);
  }
};

}  // namespace Lpm

#endif
