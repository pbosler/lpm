#ifndef LPM_VELOCITY_GALLERY_HPP
#define LPM_VELOCITY_GALLERY_HPP

#include "LpmConfig.h"
#include "lpm_geometry.hpp"
#include "lpm_constants.hpp"
#include "util/lpm_math_util.hpp"
#include "util/lpm_floating_point_util.hpp"
#include "util/lpm_tuple.hpp"
#include <cmath>
#include <memory>

namespace Lpm {

template <int ndim>
struct VelocityFunction {
  typedef std::shared_ptr<VelocityFunction> ptr;

  virtual ~VelocityFunction() {}

  virtual Tuple<Real,ndim> operator() (const Tuple<Real,ndim>& x, const Real& t) const = 0;

  std::string name() const {return std::string();}
};

// struct MovingVorticesVelocity : public VelocityFunction<3> {
//   static constexpr Real u0 = 2*constants::PI/12;
//
//   Tuple<Real,3> operator() (const Tuple<Real,3>& x, const Real& t) const {
//     const Real lat = SphereGeometry::latitude(x);
//     const Real lon = SphereGeometry::longitude(x);
//
//     const Real rho = 3*sqrt(1-square(cos(lat))*square(sin(lon-u0*t));
//
//     const Real omg =
//   }
// };

struct LauritzenEtAlDeformationalFlow : public VelocityFunction<3> {
  static constexpr Real RR = 1;
  static constexpr Real TT = 5;

  std::string name() const {return "LauritzenEtAlDeformationalFlow";}

  Tuple<Real,3> operator() (const Tuple<Real,3>& x, const Real& t) const {
    const Real lat = SphereGeometry::latitude(x);
    const Real lon = SphereGeometry::longitude(x):
    const Real u = 10 * RR/TT * square(sin(lon - 2*constants::PI*t/TT))*sin(2*lat)*cos(constants::PI*t/TT) +
      2*constants::PI*RR/TT * cos(lat);
    const Real v = 10 * RR/TT * sin(2*(lon-2*constants::PI*t/TT)) * cos(lat)*cos(constants::PI*t/TT);

    Tuple<Real,3> result;
    result[0] = -u * sin(lon) - v * sin(lat)*cos(lon);
    result[1] =  u * cos(lon) - v * sin(lat)*sin(lon);
    result[2] =  v * cos(lat);
    return result;
  }
};

struct LauritzenEtAlDivergentFlow : public VelocityFunction<3> {
  static constexpr Real RR = 1;
  static constexpr Real TT = 5;

  std::string name() const {return "LauritzenEtAlDivergentFlow";}

  Tuple<Real,3> operator() (const Tuple<Real,3>& x, const Real& t) const {
    const Real lat = SphereGeometry::latitude(x);
    const Real lon = SphereGeometry::longitude(x):
    const Real lambda_prime = lon - 2*constants::PI*t/TT;

    const Real u = -5*RR/TT*square(sin(0.5*lambda_prime))*sin(2*lat) *
      square(cos(lat))*cos(constants::PI*t/TT) + 2*constants::PI*RR/TT*cos(lat);
    const Real v = 2.5*RR/TT * sin(lambda_prime)*cube(cos(lat))*cos(constants::PI*t/TT);

    Tuple<Real,3> result;
    result[0] = -u * sin(lon) - v * sin(lat)*cos(lon);
    result[1] =  u * cos(lon) - v * sin(lat)*sin(lon);
    result[2] =  v * cos(lat);
    return result;
  }
};

struct RossbyWave54Velocity : public VelocityFunction<3> {

  std::string name() const {return "RossbyWave54Velocity";}

  Tuple<Real,3> operator() (const Tuple<Real,3>& x, const Real& t) const {

  }
};



} // namespace Lpm

#endif
