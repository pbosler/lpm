#ifndef LPM_TRACER_GALLERY_HPP
#define LPM_TRACER_GALLERY_HPP

#include "LpmConfig.h"
#include "lpm_geometry.hpp"
#include "lpm_constants.hpp"
#include "util/lpm_math.hpp"
#include "util/lpm_floating_point.hpp"
#include "util/lpm_tuple.hpp"
#include <cmath>
#include <memory>

namespace Lpm {

/**

  Eqn. (9.4) from LeVeque 1996, SIAM J. Num. Anal.
*/
struct PlanarHump {
  static constexpr Real x0 = 0.25;
  static constexpr Real y0 = 0.5;
  static constexpr Real r0 = 0.15;
  static constexpr Real h0 = 0.25;

  KOKKOS_INLINE_FUNCTION
  PlanarHump() = default;

  inline std::string name() const {return "PlanarHump";}

  template <typename CVType> KOKKOS_INLINE_FUNCTION
  Real operator() (const CVType xy) const {
    const Real rr0 = PlanarHump::r0;
    const Real xy0[2] = {PlanarHump::x0, PlanarHump::y0};
    const Real dist = PlaneGeometry::distance(xy, xy0);
    const Real r = min(dist, rr0)/rr0;
    return h0*(1 + cos(constants::PI * r));
  }

  KOKKOS_INLINE_FUNCTION
  Real operator() (const Real x, const Real y) const {
    const Real xy[2] = {x,y};
    const Real xy0[2] = {PlanarHump::x0, PlanarHump::y0};
    const Real dist = PlaneGeometry::distance(xy, xy0);
    const Real r = min(dist, PlanarHump::r0)/PlanarHump::r0;
    return h0*(1 + cos(constants::PI * r));
  }
};

struct PlanarSlottedDisk {
  static constexpr Real x0 = 0.5;
  static constexpr Real y0 = 0.75;
  static constexpr Real r0 = 0.15;
  static constexpr Real h0 = 1;

  KOKKOS_INLINE_FUNCTION
  PlanarSlottedDisk() = default;

  inline std::string name() const {return "PlanarSlottedDisk";}

  template <typename CVType> KOKKOS_INLINE_FUNCTION
  Real operator() (const CVType xy) const {
    Real result = 0;
    const Real xy0[2] = {PlanarSlottedDisk::x0, PlanarSlottedDisk::y0};
    const Real dist = PlaneGeometry::distance(xy, xy0);
    if (dist <= PlanarSlottedDisk::r0) {
      if (abs(xy(1) - y0) > PlanarSlottedDisk::r0/6) {
        result = h0;
      }
      else if (xy(0) - x0 < -5*PlanarSlottedDisk::r0/12) {
        result = h0;
      }
    }
    return result;
  }
};

struct PlanarCone {
  static constexpr Real x0 = 0.5;
  static constexpr Real y0 = 0.25;
  static constexpr Real r0 = 0.15;
  static constexpr Real h0 = 1;

  KOKKOS_INLINE_FUNCTION
  PlanarCone() = default;

  inline std::string name() const {return "PlanarCone";}

  template <typename CVType> KOKKOS_INLINE_FUNCTION
  Real operator() (const CVType xy) const {
    Real result = 0;
    const Real xy0[2] = {PlanarCone::x0, PlanarCone::y0};
    const Real dist = PlaneGeometry::distance(xy, xy0);
    if (dist <= PlanarCone::r0) {
      result = 1 - dist/PlanarCone::r0;
    }
    return result;
  }
};

struct SphericalSlottedCylinders {
  static constexpr Real lat1 = 0;
  static constexpr Real lon1 = 5*constants::PI/6;
  static constexpr Real lat2 = 0;
  static constexpr Real lon2 = 7*constants::PI/6;
  static constexpr Real RR = 0.5;
  static constexpr Real b = 0.1;
  static constexpr Real c = 1;

  KOKKOS_INLINE_FUNCTION
  SphericalSlottedCylinders() = default;

  inline std::string name() const  {return "SphericalSlottedCylinders";}

  template <typename CVType> KOKKOS_INLINE_FUNCTION
  Real operator() (const CVType xyz) const {
    Real result = b;
    const Real xyz_ctr1[3] = {std::cos(lon1)*std::cos(lat1),
                              std::sin(lon1)*std::cos(lat1),
                              std::sin(lat1)};
    const Real xyz_ctr2[3] = {std::cos(lon2)*std::cos(lat2),
                              std::sin(lon2)*std::cos(lat2),
                              std::sin (lat2)};
    const Real lat = SphereGeometry::latitude(xyz);
    const Real lon = SphereGeometry::longitude(xyz);
    const Real r1 = SphereGeometry::distance(xyz, xyz_ctr1);
    const Real r2 = SphereGeometry::distance(xyz, xyz_ctr2);

    if (r1 <= RR) {
      if (abs(lon - lon1) >= RR/6) {
        result = c;
      }
      else if (lat - lat1 < -5*RR/12) {
        result = c;
      }
    }

    if (r2 <= RR) {
      if (abs(lon-lon2) >= RR/6) {
        result = c;
      }
      else if (lat - lat2 > 5*RR/12) {
        result = c;
      }
    }

    return result;
  }

  KOKKOS_INLINE_FUNCTION
  Real operator() (const Real& x, const Real& y, const Real& z) const  {
    Real result = b;
    const Real xyz[3] = {x, y, z};
    const Real xyz_ctr1[3] = {std::cos(lon1)*std::cos(lat1),
                              std::sin(lon1)*std::cos(lat1),
                              std::sin(lat1)};
    const Real xyz_ctr2[3] = {std::cos(lon2)*std::cos(lat2),
                              std::sin(lon2)*std::cos(lat2),
                              std::sin (lat2)};
    const Real lat = SphereGeometry::latitude(xyz);
    const Real lon = SphereGeometry::longitude(xyz);
    const Real r1 = SphereGeometry::distance(xyz, xyz_ctr1);
    const Real r2 = SphereGeometry::distance(xyz, xyz_ctr2);

    if (r1 <= RR) {
      if (abs(lon - lon1) >= RR/6) {
        result = c;
      }
      else if (lat - lat1 < -5*RR/12) {
        result = c;
      }
    }

    if (r2 <= RR) {
      if (abs(lon-lon2) >= RR/6) {
        result = c;
      }
      else if (lat - lat2 > 5*RR/12) {
        result = c;
      }
    }

    return result;
  }
};

struct SphericalCosineBells {
  static constexpr Real lat1 = 0;
  static constexpr Real lon1 = 5*constants::PI/6;
  static constexpr Real lat2 = 0;
  static constexpr Real lon2 = 7*constants::PI/6;
  static constexpr Real RR = 0.5;
  static constexpr Real b = 0.1;
  static constexpr Real c = 0.9;
  static constexpr Real hmax = 1;

  KOKKOS_INLINE_FUNCTION
  SphericalCosineBells() = default;

  std::string name() const {return "SphericalCosineBells";}

  template <typename CVType> KOKKOS_INLINE_FUNCTION
  Real operator() (const CVType xyz) const {
    Real result = 0;
    const Real xyz_ctr1[3] = {std::cos(lon1)*std::cos(lat1),
                              std::sin(lon1)*std::cos(lat1),
                              std::sin(lat1)};
    const Real xyz_ctr2[3] = {std::cos(lon2)*std::cos(lat2),
                              std::sin(lon2)*std::cos(lat2),
                              std::sin (lat2)};
    const Real r1 = SphereGeometry::distance(xyz, xyz_ctr1);
    const Real r2 = SphereGeometry::distance(xyz, xyz_ctr2);

    const Real h1 = 0.5*hmax*(1 + cos(constants::PI*r1/RR));
    const Real h2 = 0.5*hmax*(1 + cos(constants::PI*r2/RR));

    if (r1 <= RR) {
      result += c*h1;
    }

    if (r2 <= RR) {
      result += c*h2;
    }

    return result;
  }

  KOKKOS_INLINE_FUNCTION
  Real operator() (const Real& x, const Real& y, const Real& z) const {
    Real result = b;
    const Real xyz[3] = {x, y, z};
    const Real xyz_ctr1[3] = {std::cos(lon1)*std::cos(lat1),
                              std::sin(lon1)*std::cos(lat1),
                              std::sin(lat1)};
    const Real xyz_ctr2[3] = {std::cos(lon2)*std::cos(lat2),
                              std::sin(lon2)*std::cos(lat2),
                              std::sin (lat2)};
    const Real r1 = SphereGeometry::distance(xyz, xyz_ctr1);
    const Real r2 = SphereGeometry::distance(xyz, xyz_ctr2);

    const Real h1 = 0.5*hmax*(1 + cos(constants::PI*r1/RR));
    const Real h2 = 0.5*hmax*(1 + cos(constants::PI*r2/RR));

    if (r1 <= RR) {
      result += c*h1;
    }

    if (r2 <= RR) {
      result += c*h2;
    }

    return result;
  }
};

struct SphericalGaussianHills {
  static constexpr Real lat1 = 0;
  static constexpr Real lon1 = 5*constants::PI/6;
  static constexpr Real lat2 = 0;
  static constexpr Real lon2 = 7*constants::PI/6;
  static constexpr Real beta = 5;
  static constexpr Real hmax = 0.95;

  KOKKOS_INLINE_FUNCTION
  SphericalGaussianHills() = default;

  std::string name() const {return "SphericalGaussianHills";}

  template <typename CVType> KOKKOS_INLINE_FUNCTION
  Real operator() (const CVType xyz) const {
    const Real xyz_ctr1[3] = {std::cos(lon1)*std::cos(lat1),
                              std::sin(lon1)*std::cos(lat1),
                              std::sin(lat1)};
    const Real xyz_ctr2[3] = {std::cos(lon2)*std::cos(lat2),
                              std::sin(lon2)*std::cos(lat2),
                              std::sin (lat2)};

    const Real sqdist1 = SphereGeometry::square_euclidean_distance(xyz, xyz_ctr1);
    const Real sqdist2 = SphereGeometry::square_euclidean_distance(xyz, xyz_ctr2);

    const Real h1 = hmax*std::exp(-beta*sqdist1);
    const Real h2 = hmax*std::exp(-beta*sqdist2);

    return h1 + h2;
  }

  KOKKOS_INLINE_FUNCTION
  Real operator() (const Real& x, const Real& y, const Real& z) const {
    const Real xyz[3] = {x, y, z};
    const Real xyz_ctr1[3] = {std::cos(lon1)*std::cos(lat1),
                              std::sin(lon1)*std::cos(lat1),
                              std::sin(lat1)};
    const Real xyz_ctr2[3] = {std::cos(lon2)*std::cos(lat2),
                              std::sin(lon2)*std::cos(lat2),
                              std::sin (lat2)};

    const Real sqdist1 = SphereGeometry::square_euclidean_distance(xyz, xyz_ctr1);
    const Real sqdist2 = SphereGeometry::square_euclidean_distance(xyz, xyz_ctr2);

    const Real h1 = hmax*std::exp(-beta*sqdist1);
    const Real h2 = hmax*std::exp(-beta*sqdist2);

    return h1 + h2;
  }
};

struct MovingVorticesTracer {
  static constexpr Real u0 = 2*constants::PI/12;

  KOKKOS_INLINE_FUNCTION
  MovingVorticesTracer() = default;

  inline std::string name() const {return "MovingVorticesTracer";}

  template <typename CVType> KOKKOS_INLINE_FUNCTION
  Real operator() (const CVType xyz, const Real t=0) const {
    const Real lat = SphereGeometry::latitude(xyz);
    const Real lon = SphereGeometry::longitude(xyz);

    const Real lambda_prime = atan4(-std::cos(lon - u0*t), std::tan(lat));

    const Real rho = 3 * sqrt(1-square(cos(lat))*square(sin(lon-u0*t)));
    const Real omg = 1.5*sqrt(3.0)*u0*std::tanh(rho)*FloatingPoint<Real>::safe_denominator(std::cosh(rho));

    return 1 - std::tanh(0.2*rho*std::sin(lambda_prime-omg*t));
  }

  KOKKOS_INLINE_FUNCTION
  Real operator() (const Real& x, const Real& y, const Real& z, const Real& t=0) const {
    const Real xyz[3] = {x, y, z};
    const Real lat = SphereGeometry::latitude(xyz);
    const Real lon = SphereGeometry::longitude(xyz);

    const Real lambda_prime = atan4(-std::cos(lon - u0*t), std::tan(lat));

    const Real rho = 3 * sqrt(1-square(cos(lat))*square(sin(lon-u0*t)));
    const Real omg = 1.5*sqrt(3.0)*u0*std::tanh(rho)*FloatingPoint<Real>::safe_denominator(std::cosh(rho));

    return 1 - std::tanh(0.2*rho*std::sin(lambda_prime-omg*t));
  }
};

struct LatitudeTracer {
  KOKKOS_INLINE_FUNCTION
  LatitudeTracer() = default;

  inline std::string name() const {return "LatitudeTracer";}

  template <typename CVType> KOKKOS_INLINE_FUNCTION
  Real operator() (const CVType xyz) const {
    return SphereGeometry::latitude(xyz);
  }

  KOKKOS_INLINE_FUNCTION
  Real operator() (const Real& x, const Real& y, const Real& z) const {
    const Real xyz[3] = {x,y,z};
    return SphereGeometry::latitude(xyz);
  }
};


} // namespace Lpm

#endif
