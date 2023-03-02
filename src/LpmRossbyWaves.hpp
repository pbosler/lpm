#ifndef LPM_ROSSBY_WAVES_HPP
#define LPM_ROSSBY_WAVES_HPP

#include <cmath>

#include "Kokkos_Core.hpp"
#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmKokkosUtil.hpp"
#include "LpmUtilities.hpp"

namespace Lpm {

/** @brief Associated Legendre polynomial \f$ P_5^4(x) \f$
  @param z input value
*/
KOKKOS_INLINE_FUNCTION
Real legendre54(const Real& z) { return z * (z * z - 1.0) * (z * z - 1.0); }

/** @brief Spherical Harmonic \f$ Y_5^4(x) \f$

  @param x input coordinate vector
*/
template <typename VecType>
KOKKOS_INLINE_FUNCTION Real SphHarm54(const VecType& x) {
  const Real lam = SphereGeometry::longitude(x);
  return 30 * std::cos(4 * lam) * legendre54(x[2]);
}

/** @brief If vorticity is defined by SphHarm54(), then the resulting velocity
  is given by this function (BVE Only, not SWE).

  @param x input coordinate vector
*/
template <typename VT>
KOKKOS_INLINE_FUNCTION ko::Tuple<Real, 3> RH54Velocity(const VT& x) {
  const Real theta = SphereGeometry::latitude(x);
  const Real lambda = SphereGeometry::longitude(x);
  const Real coslat = std::cos(theta);
  const Real sinlat = std::sin(theta);
  const Real coslon = std::cos(lambda);
  const Real sinlon = std::sin(lambda);
  const Real usph =
      0.5 * std::cos(4 * lambda) * cube(coslat) * (5 * std::cos(2 * theta) - 3);
  const Real vsph = 4 * cube(coslat) * sinlat * std::sin(4 * lambda);
  const Real u = -usph * sinlon - vsph * sinlat * coslon;
  const Real v = usph * coslon - vsph * sinlat * sinlon;
  const Real w = vsph * coslat;
  return ko::Tuple<Real, 3>(u, v, w);
}

}  // namespace Lpm
#endif
