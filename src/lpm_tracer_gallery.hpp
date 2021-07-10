#ifndef LPM_TRACER_GALLERY_HPP
#define LPM_TRACER_GALLERY_HPP

#include "LpmConfig.h"
#include "lpm_geometry.hpp"
#include "lpm_constants.hpp"
#include "util/lpm_math_util.hpp"
#include "util/lpm_floating_point_util.hpp"
#include "util/lpm_tuple.hpp"
#include <cmath>
#include <memory>

namespace Lpm {

struct TracerInitialCondition {
  typedef std::shared_ptr<TracerInitialCondition> ptr;

  virtual ~TracerInitialCondition() {}

  virtual Real operator() (const Real& x, const Real& y) const = 0;

  virtual Real operator() (const Real& x, const Real& y, const Real& z) const = 0;

  virtual std::string name() const {return std::string();}
};

struct SphericalSlottedCylinders : public TracerInitialCondition {
  static constexpr Real lat1 = 0;
  static constexpr Real lon1 = 5*constants::PI/6;
  static constexpr Real lat2 = 0;
  static constexpr Real lon2 = 7*constants::PI/6;
  static constexpr Real RR = 0.5;
  static constexpr Real b = 0.1;
  static constexpr Real c = 1;

  inline std::string name() const {return "SphericalSlottedCylinders";}

  inline Real operator() (const Real& x, const Real& y) const {return 0;}

  inline Real operator() (const Real& x, const Real& y, const Real& z) const {
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

struct SphericalCosineBells : public TracerInitialCondition {
  static constexpr Real lat1 = 0;
  static constexpr Real lon1 = 5*constants::PI/6;
  static constexpr Real lat2 = 0;
  static constexpr Real lon2 = 7*constants::PI/6;
  static constexpr Real RR = 0.5;
  static constexpr Real b = 0.1;
  static constexpr Real c = 0.9;
  static constexpr Real hmax = 1;

  std::string name() const {return "SphericalCosineBells";}

  inline Real operator() (const Real& x, const Real& y) const {return 0;}

  inline Real operator() (const Real& x, const Real& y, const Real& z) const {
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

struct SphericalGaussianHills : public TracerInitialCondition {
  static constexpr Real lat1 = 0;
  static constexpr Real lon1 = 5*constants::PI/6;
  static constexpr Real lat2 = 0;
  static constexpr Real lon2 = 7*constants::PI/6;
  static constexpr Real beta = 5;
  static constexpr Real hmax = 0.95;

  std::string name() const {return "SphericalGaussianHills";}

  inline Real operator() (const Real& x, const Real& y) const {return 0;}

  inline Real operator() (const Real& x, const Real& y, const Real& z) const {
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

struct MovingVorticesTracer : public TracerInitialCondition {
  static constexpr Real u0 = 2*constants::PI/12;

  inline std::string name() const {return "MovingVorticesTracer";}

  inline Real operator() (const Real& x, const Real& y) const {return 0;}

  inline Real operator() (const Real& x, const Real& y, const Real& z) const {
    return this->(x,y,z,0);
  }

  inline Real operator() (const Real& x, const Real& y, const Real& z const Real & t) const {
    const Real xyz[3] = {x, y, z};
    const Real lat = SphereGeometry::latitude(xyz);
    const Real lon = SphereGeometry::longitude(xyz);

    const Real lambda_prime = atan4(-std::cos(lon - u0*t), std::tan(lat));

    const Real rho = 3 * sqrt(1-square(cos(lat))*square(sin(lon-u0*t));
    const Real omg = 1.5*sqrt(3.0)*u0*std::tanh(rho)*FloatingPoint<Real>::safe_denom(std::cosh(rho));

    return 1 - std::tanh(0.2*rho*std::sin(lambda_prime-omg*t));
  }
};

struct LatitudeTracer : public TracerInitialCondition {
  inline std::string name() const {return "LatitudeTracer";}

  inline Real operator() (const Real& x, const Real& y) const {return 0;}

  inline Real operator() (const Real& x, const Real& y, const Real& z) const {
    const Real xyz[3] = {x,y,z};
    return SphereGeometry::latitude(xyz);
  }
};


} // namespace Lpm

#endif
