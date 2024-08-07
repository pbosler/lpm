#ifndef LPM_VELOCITY_GALLERY_HPP
#define LPM_VELOCITY_GALLERY_HPP

#include <cmath>
#include <memory>

#include "LpmConfig.h"
#include "lpm_constants.hpp"
#include "lpm_geometry.hpp"
#include "lpm_vorticity_gallery.hpp"
#include "util/lpm_floating_point.hpp"
#include "util/lpm_math.hpp"
#include "util/lpm_tuple.hpp"

namespace Lpm {
using Kokkos::Tuple;

/** Struct to apply a velocity functor

  Any of the velocity functions below can be given as template parameters to this
  functor and they will be evaluated in parallel at each coordinate.
*/
template <typename VelocityFtor>
struct VelocityKernel {
  Kokkos::View<Real**> velocity;
  Kokkos::View<Real**> xcrds;
  Real t;
  VelocityFtor velfn;

  VelocityKernel(Kokkos::View<Real**> u, const Kokkos::View<Real**> x,
                 const Real tt)
      : velocity(u), xcrds(x), t(tt) {}

  VelocityKernel(Kokkos::View<Real**> u, const Kokkos::View<Real**> x,
    const Real tt, const VelocityFtor& vel_fn) :
      velocity(u), xcrds(x), t(tt), velfn(vel_fn) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index i) const {
    const auto myx = Kokkos::subview(xcrds, i, Kokkos::ALL);
    auto myu = Kokkos::subview(velocity, i, Kokkos::ALL);
    Kokkos::Tuple<Real, VelocityFtor::ndim> u = velfn(myx, t);
    for (int j = 0; j < VelocityFtor::ndim; ++j) {
      myu(j) = u[j];
    }
  }
};

struct PlanarConstantEastward {
  typedef PlaneGeometry geo;
  static constexpr Int ndim = 2;
  static constexpr Real u0 = 8;

  KOKKOS_INLINE_FUNCTION
  PlanarConstantEastward() = default;

  inline std::string name() const { return "PlanarConstantEastward"; }

  template <typename CV>
  KOKKOS_INLINE_FUNCTION Tuple<Real, 2> operator()(const CV xy,
                                                   const Real t = 0) const {
    Tuple<Real, 2> result;
    result[0] = u0;
    result[1] = 0;
    return result;
  }
};

struct PlanarRigidRotation {
  typedef PlaneGeometry geo;
  static constexpr Int ndim = 2;
  static constexpr Real Omega = 2 * constants::PI / 5;
  static constexpr Real x0 = 0;
  static constexpr Real y0 = 0;

  KOKKOS_INLINE_FUNCTION
  PlanarRigidRotation() = default;

  inline std::string name() const { return "PlanarRigidRotation"; }

  template <typename CV>
  KOKKOS_INLINE_FUNCTION Tuple<Real, 2> operator()(const CV xy,
                                                   const Real t = 0) const {
    Tuple<Real, 2> result;
    result[0] = -Omega * (xy[1] - x0);
    result[1] = Omega * (xy[0] - y0);
    return result;
  }
};

struct PlanarDeformationalFlow {
  typedef PlaneGeometry geo;
  static constexpr Int ndim = 2;
  static constexpr Real TT = 5;

  KOKKOS_INLINE_FUNCTION
  PlanarDeformationalFlow() = default;

  inline std::string name() const { return "PlanarDeformationalFlow"; }

  template <typename CV>
  KOKKOS_INLINE_FUNCTION Tuple<Real, 2> operator()(const CV xy,
                                                   const Real t = 0) const {
    Tuple<Real, 2> result;
    const Real gt = cos(constants::PI * t / TT);
    result[0] = square(sin(constants::PI * xy[0])) *
                sin(2 * constants::PI * xy[1]) * gt;
    result[1] = -square(sin(constants::PI * xy[1])) *
                sin(2 * constants::PI * xy[0]) * gt;
    return result;
  }
};

struct SphericalRigidRotation {
  typedef SphereGeometry geo;
  static constexpr Int ndim = 3;
  static constexpr Real alpha = 0;
  static constexpr Real u0 = 2 * constants::PI / 5;

  KOKKOS_INLINE_FUNCTION
  SphericalRigidRotation() = default;

  inline std::string name() const { return "SphericalRigidRotation"; }

  template <typename CV>
  KOKKOS_INLINE_FUNCTION Tuple<Real, 3> operator()(const CV xyz,
                                                   const Real t = 0) const {
    const Real lat = SphereGeometry::latitude(xyz);
    const Real lon = SphereGeometry::longitude(xyz);
    const Real u =
        u0 * (cos(lat) * cos(alpha) + sin(lat) * cos(lon) * sin(alpha));
    const Real v = -u0 * sin(lon) * sin(alpha);

    Tuple<Real, 3> result;
    result[0] = -u * sin(lon) - v * sin(lat) * cos(lon);
    result[1] = u * cos(lon) - v * sin(lat) * sin(lon);
    result[2] = v * cos(lat);
    return result;
  }
};

/** Velocity field from the Nair and Jablonowski (2006) moving vortices test.
*/
struct MovingVorticesVelocity {
  typedef SphereGeometry geo;
  static constexpr Int ndim = 3;
  static constexpr Real u0 = 2 * constants::PI / 12;

  KOKKOS_INLINE_FUNCTION
  MovingVorticesVelocity() = default;

  inline std::string name() const { return "SphericalMovingVortices"; }

  template <typename CV>
  KOKKOS_INLINE_FUNCTION Tuple<Real, 3> operator()(const CV& x,
                                                   const Real& t) const {
    const Real lat = SphereGeometry::latitude(x);
    const Real lon = SphereGeometry::longitude(x);
    const Real lon_prime = lon - u0 * t;
    const Real rho = 3 * sqrt(1 - square(cos(lat)) * square(sin(lon_prime)));
    const Real omg = 1.5 * sqrt(3.0) * u0 * tanh(rho) *
                     FloatingPoint<Real>::safe_denominator(rho) /
                     square(cosh(rho));

    const Real u = omg * sin(lon_prime) * sin(lat) + u0 * cos(lat);
    const Real v = omg * cos(lon_prime);

    Tuple<Real, 3> result;
    result[0] = -u * sin(lon) - v * sin(lat) * cos(lon);
    result[1] = u * cos(lon) - v * sin(lat) * sin(lon);
    result[2] = v * cos(lat);
    return result;
  }
};

/** Velocity field from the Lauritzen et al. (2012) test cases.
*/
struct LauritzenEtAlDeformationalFlow {
  typedef SphereGeometry geo;
  static constexpr Int ndim = 3;
  static constexpr Real RR = 1;
  static constexpr Real TT = 5;

  KOKKOS_INLINE_FUNCTION
  LauritzenEtAlDeformationalFlow() = default;

  std::string name() const { return "LauritzenEtAlDeformationalFlow"; }

  template <typename CV>
  KOKKOS_INLINE_FUNCTION Tuple<Real, 3> operator()(const CV x,
                                                   const Real& t) const {
    const Real lat = SphereGeometry::latitude(x);
    const Real lon = SphereGeometry::longitude(x);
    const Real u = 10 * RR / TT *
                       square(sin(lon - 2 * constants::PI * t / TT)) *
                       sin(2 * lat) * cos(constants::PI * t / TT) +
                   2 * constants::PI * RR / TT * cos(lat);
    const Real v = 10 * RR / TT * sin(2 * (lon - 2 * constants::PI * t / TT)) *
                   cos(lat) * cos(constants::PI * t / TT);

    Tuple<Real, 3> result;
    result[0] = -u * sin(lon) - v * sin(lat) * cos(lon);
    result[1] = u * cos(lon) - v * sin(lat) * sin(lon);
    result[2] = v * cos(lat);
    return result;
  }
};

/** Velocity field from the Lauritzen et al. (2012) test cases
  with nonzero divergence.
*/
struct LauritzenEtAlDivergentFlow {
  typedef SphereGeometry geo;
  static constexpr Int ndim = 3;
  static constexpr Real RR = 1;
  static constexpr Real TT = 5;

  KOKKOS_INLINE_FUNCTION
  LauritzenEtAlDivergentFlow() = default;

  std::string name() const { return "LauritzenEtAlDivergentFlow"; }

  template <typename CV>
  KOKKOS_INLINE_FUNCTION Tuple<Real, 3> operator()(const CV x,
                                                   const Real& t) const {
    const Real lat = SphereGeometry::latitude(x);
    const Real lon = SphereGeometry::longitude(x);
    const Real lambda_prime = lon - 2 * constants::PI * t / TT;

    const Real u = -5 * RR / TT * square(sin(0.5 * lambda_prime)) *
                       sin(2 * lat) * square(cos(lat)) *
                       cos(constants::PI * t / TT) +
                   2 * constants::PI * RR / TT * cos(lat);
    const Real v = 2.5 * RR / TT * sin(lambda_prime) * cube(cos(lat)) *
                   cos(constants::PI * t / TT);

    Tuple<Real, 3> result;
    result[0] = -u * sin(lon) - v * sin(lat) * cos(lon);
    result[1] = u * cos(lon) - v * sin(lat) * sin(lon);
    result[2] = v * cos(lat);
    return result;
  }
};

/** Velocity field generated by the Y_5^4 spherical harmonic.


*/
// TODO: account for time (wave advection)
struct RossbyWave54Velocity {
  typedef SphereGeometry geo;
  static constexpr Int ndim = 3;
  Real background_rotation;
  Real rh54_amplitude;

  KOKKOS_INLINE_FUNCTION
  RossbyWave54Velocity(const Real u0 = 0, const Real amp=1) :
    background_rotation(u0), rh54_amplitude(amp) {}

  KOKKOS_INLINE_FUNCTION
  RossbyWave54Velocity(const RossbyHaurwitz54& rh54_vorticity) :
    background_rotation(rh54_vorticity.u0),
    rh54_amplitude(rh54_vorticity.rh54_amplitude) {}

  KOKKOS_INLINE_FUNCTION
  RossbyWave54Velocity(const RossbyWave54Velocity& other) = default;

  std::string name() const { return "RossbyWave54Velocity"; }

  template <typename CV>
  KOKKOS_INLINE_FUNCTION Tuple<Real, 3> operator()(const CV x,
                                                   const Real& t) const {
    const Real lat = SphereGeometry::latitude(x);
    const Real lon = SphereGeometry::longitude(x);

    const Real u = rh54_amplitude * 0.5 * cos(4 * lon) * cube(cos(lat)) * (5 * cos(2 * lat) - 3);
    const Real v = rh54_amplitude * 4 * cube(cos(lat)) * sin(lat) * sin(4 * lon);
    Tuple<Real, 3> result;
    result[0] =
        -background_rotation * x(1) - u * sin(lon) - v * sin(lat) * cos(lon);
    result[1] =
        background_rotation * x(0) + u * cos(lon) - v * sin(lat) * sin(lon);
    result[2] = v * cos(lat);
    return result;
  }
};

}  // namespace Lpm

#endif
