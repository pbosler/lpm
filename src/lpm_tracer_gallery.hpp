#ifndef LPM_TRACER_GALLERY_HPP
#define LPM_TRACER_GALLERY_HPP

#include <cmath>
#include <memory>

#include "LpmConfig.h"
#include "lpm_constants.hpp"
#include "lpm_geometry.hpp"
#include "util/lpm_floating_point.hpp"
#include "util/lpm_math.hpp"
#include "util/lpm_tuple.hpp"

namespace Lpm {

template <typename Geo, typename TracerType>
struct LagrangianTracerKernel {
  using crd_view = typename Geo::crd_view_type;
  scalar_view_type tracer_vals;
  crd_view lag_crds;
  TracerType tracer;

  LagrangianTracerKernel(scalar_view_type vals, const crd_view& lcrds,
    const TracerType& tr) :
    tracer_vals(vals),
    lag_crds(lcrds),
    tracer(tr) {}

    KOKKOS_INLINE_FUNCTION
    void operator() (const Index i) const {
      const auto mcrd = Kokkos::subview(lag_crds, i, Kokkos::ALL);
      tracer_vals(i) = tracer(mcrd);
    }
};

struct PlanarGaussian {
  typedef PlaneGeometry geo;
  static constexpr Real b = 1;
  static constexpr bool IsVorticity = false;

  KOKKOS_INLINE_FUNCTION
  PlanarGaussian() = default;

  inline std::string name() const { return "PlanarGaussian"; }

  template <typename CVT>
  KOKKOS_INLINE_FUNCTION Real operator()(const CVT xy) const {
    const Real rsq = PlaneGeometry::norm2(xy);
    return exp(-PlanarGaussian::b * rsq);
  }

  template <typename CVT>
  KOKKOS_INLINE_FUNCTION Real laplacian(const CVT xy) const {
    const Real rsq = PlaneGeometry::norm2(xy);
    const Real b = PlanarGaussian::b;
    return 4 * b * exp(-b * rsq) * (square(b) * rsq - 1);
  }
};

struct PlanarRings {
  typedef PlaneGeometry geo;
  static constexpr bool IsVorticity = false;

  KOKKOS_INLINE_FUNCTION
  PlanarRings() = default;

  inline std::string name() const { return "PlanarRings"; }

  template <typename CVType>
  KOKKOS_INLINE_FUNCTION Real operator()(const CVType xy) const {
    const Real r = geo::mag(xy);
    return r * exp(-r) * sin(r);
  }

  template <typename CVType>
  KOKKOS_INLINE_FUNCTION Real laplacian(const CVType xy) const {
    const Real r = geo::mag(xy);
    return safe_divide(r) * (exp(-r) * ((-2 * square(r) + 3 * r) * cos(r) +
                                        (1 - 3 * r) * sin(r)));
  }
};

/**

  Eqn. (9.4) from LeVeque 1996, SIAM J. Num. Anal.
*/
struct PlanarHump {
  typedef PlaneGeometry geo;
  static constexpr bool IsVorticity = false;
  static constexpr Real x0 = 0.25;
  static constexpr Real y0 = 0.5;
  static constexpr Real r0 = 0.15;
  static constexpr Real h0 = 0.25;

  KOKKOS_INLINE_FUNCTION
  PlanarHump() = default;

  inline std::string name() const { return "PlanarHump"; }

  template <typename CVType>
  KOKKOS_INLINE_FUNCTION Real operator()(const CVType xy) const {
    const Real rr0 = PlanarHump::r0;
    const Real xy0[2] = {PlanarHump::x0, PlanarHump::y0};
    const Real dist = PlaneGeometry::distance(xy, xy0);
    const Real r = min(dist, rr0) / rr0;
    return h0 * (1 + cos(constants::PI * r));
  }
};

struct PlanarSlottedDisk {
  typedef PlaneGeometry geo;
  static constexpr bool IsVorticity = false;
  static constexpr Real x0 = 0.5;
  static constexpr Real y0 = 0.75;
  static constexpr Real r0 = 0.15;
  static constexpr Real h0 = 1;

  KOKKOS_INLINE_FUNCTION
  PlanarSlottedDisk() = default;

  inline std::string name() const { return "PlanarSlottedDisk"; }

  template <typename CVType>
  KOKKOS_INLINE_FUNCTION Real operator()(const CVType xy) const {
    Real result = 0;
    const Real xy0[2] = {PlanarSlottedDisk::x0, PlanarSlottedDisk::y0};
    const Real dist = PlaneGeometry::distance(xy, xy0);
    if (dist <= PlanarSlottedDisk::r0) {
      if (abs(xy(1) - y0) > PlanarSlottedDisk::r0 / 6) {
        result = h0;
      } else if (xy(0) - x0 < -5 * PlanarSlottedDisk::r0 / 12) {
        result = h0;
      }
    }
    return result;
  }
};

struct PlanarCone {
  typedef PlaneGeometry geo;
  static constexpr bool IsVorticity = false;
  static constexpr Real x0 = 0.5;
  static constexpr Real y0 = 0.25;
  static constexpr Real r0 = 0.15;
  static constexpr Real h0 = 1;

  KOKKOS_INLINE_FUNCTION
  PlanarCone() = default;

  inline std::string name() const { return "PlanarCone"; }

  template <typename CVType>
  KOKKOS_INLINE_FUNCTION Real operator()(const CVType xy) const {
    Real result = 0;
    const Real xy0[2] = {PlanarCone::x0, PlanarCone::y0};
    const Real dist = PlaneGeometry::distance(xy, xy0);
    if (dist <= PlanarCone::r0) {
      result = 1 - dist / PlanarCone::r0;
    }
    return result;
  }
};

struct SphericalSlottedCylinders {
  typedef SphereGeometry geo;
  static constexpr bool IsVorticity = false;
  static constexpr Real lat1 = 0;
  static constexpr Real lon1 = 5 * constants::PI / 6;
  static constexpr Real lat2 = 0;
  static constexpr Real lon2 = 7 * constants::PI / 6;
  static constexpr Real RR = 0.5;
  static constexpr Real b = 0.1;
  static constexpr Real c = 1;

  KOKKOS_INLINE_FUNCTION
  SphericalSlottedCylinders() = default;

  inline std::string name() const { return "SphericalSlottedCylinders"; }

  template <typename CVType>
  KOKKOS_INLINE_FUNCTION Real operator()(const CVType xyz) const {
    Real result = b;
    const Real xyz_ctr1[3] = {std::cos(lon1) * std::cos(lat1),
                              std::sin(lon1) * std::cos(lat1), std::sin(lat1)};
    const Real xyz_ctr2[3] = {std::cos(lon2) * std::cos(lat2),
                              std::sin(lon2) * std::cos(lat2), std::sin(lat2)};
    const Real lat = SphereGeometry::latitude(xyz);
    const Real lon = SphereGeometry::longitude(xyz);
    const Real r1 = SphereGeometry::distance(xyz, xyz_ctr1);
    const Real r2 = SphereGeometry::distance(xyz, xyz_ctr2);

    if (r1 <= RR) {
      if (abs(lon - lon1) >= RR / 6) {
        result = c;
      } else if (lat - lat1 < -5 * RR / 12) {
        result = c;
      }
    }

    if (r2 <= RR) {
      if (abs(lon - lon2) >= RR / 6) {
        result = c;
      } else if (lat - lat2 > 5 * RR / 12) {
        result = c;
      }
    }

    return result;
  }

  KOKKOS_INLINE_FUNCTION
  Real operator()(const Real& x, const Real& y, const Real& z) const {
    Real result = b;
    const Real xyz[3] = {x, y, z};
    const Real xyz_ctr1[3] = {std::cos(lon1) * std::cos(lat1),
                              std::sin(lon1) * std::cos(lat1), std::sin(lat1)};
    const Real xyz_ctr2[3] = {std::cos(lon2) * std::cos(lat2),
                              std::sin(lon2) * std::cos(lat2), std::sin(lat2)};
    const Real lat = SphereGeometry::latitude(xyz);
    const Real lon = SphereGeometry::longitude(xyz);
    const Real r1 = SphereGeometry::distance(xyz, xyz_ctr1);
    const Real r2 = SphereGeometry::distance(xyz, xyz_ctr2);

    if (r1 <= RR) {
      if (abs(lon - lon1) >= RR / 6) {
        result = c;
      } else if (lat - lat1 < -5 * RR / 12) {
        result = c;
      }
    }

    if (r2 <= RR) {
      if (abs(lon - lon2) >= RR / 6) {
        result = c;
      } else if (lat - lat2 > 5 * RR / 12) {
        result = c;
      }
    }

    return result;
  }
};

struct SphericalCosineBells {
  typedef SphereGeometry geo;
  static constexpr bool IsVorticity = false;
  static constexpr Real lat1 = 0;
  static constexpr Real lon1 = 5 * constants::PI / 6;
  static constexpr Real lat2 = 0;
  static constexpr Real lon2 = 7 * constants::PI / 6;
  static constexpr Real RR = 0.5;
  static constexpr Real b = 0.1;
  static constexpr Real c = 0.9;
  static constexpr Real hmax = 1;

  KOKKOS_INLINE_FUNCTION
  SphericalCosineBells() = default;

  std::string name() const { return "SphericalCosineBells"; }

  template <typename CVType>
  KOKKOS_INLINE_FUNCTION Real operator()(const CVType xyz) const {
    Real result = 0;
    const Real xyz_ctr1[3] = {std::cos(lon1) * std::cos(lat1),
                              std::sin(lon1) * std::cos(lat1), std::sin(lat1)};
    const Real xyz_ctr2[3] = {std::cos(lon2) * std::cos(lat2),
                              std::sin(lon2) * std::cos(lat2), std::sin(lat2)};
    const Real r1 = SphereGeometry::distance(xyz, xyz_ctr1);
    const Real r2 = SphereGeometry::distance(xyz, xyz_ctr2);

    const Real h1 = 0.5 * hmax * (1 + cos(constants::PI * r1 / RR));
    const Real h2 = 0.5 * hmax * (1 + cos(constants::PI * r2 / RR));

    if (r1 <= RR) {
      result += c * h1;
    }

    if (r2 <= RR) {
      result += c * h2;
    }

    return result;
  }

  KOKKOS_INLINE_FUNCTION
  Real operator()(const Real& x, const Real& y, const Real& z) const {
    Real result = b;
    const Real xyz[3] = {x, y, z};
    const Real xyz_ctr1[3] = {std::cos(lon1) * std::cos(lat1),
                              std::sin(lon1) * std::cos(lat1), std::sin(lat1)};
    const Real xyz_ctr2[3] = {std::cos(lon2) * std::cos(lat2),
                              std::sin(lon2) * std::cos(lat2), std::sin(lat2)};
    const Real r1 = SphereGeometry::distance(xyz, xyz_ctr1);
    const Real r2 = SphereGeometry::distance(xyz, xyz_ctr2);

    const Real h1 = 0.5 * hmax * (1 + cos(constants::PI * r1 / RR));
    const Real h2 = 0.5 * hmax * (1 + cos(constants::PI * r2 / RR));

    if (r1 <= RR) {
      result += c * h1;
    }

    if (r2 <= RR) {
      result += c * h2;
    }

    return result;
  }
};

struct SphericalGaussianHills {
  typedef SphereGeometry geo;
  static constexpr bool IsVorticity = false;
  static constexpr Real lat1 = 0;
  static constexpr Real lon1 = 5 * constants::PI / 6;
  static constexpr Real lat2 = 0;
  static constexpr Real lon2 = 7 * constants::PI / 6;
  static constexpr Real beta = 5;
  static constexpr Real hmax = 0.95;

  KOKKOS_INLINE_FUNCTION
  SphericalGaussianHills() = default;

  std::string name() const { return "SphericalGaussianHills"; }

  template <typename CVType>
  KOKKOS_INLINE_FUNCTION Real operator()(const CVType xyz) const {
    const Real xyz_ctr1[3] = {std::cos(lon1) * std::cos(lat1),
                              std::sin(lon1) * std::cos(lat1), std::sin(lat1)};
    const Real xyz_ctr2[3] = {std::cos(lon2) * std::cos(lat2),
                              std::sin(lon2) * std::cos(lat2), std::sin(lat2)};

    const Real sqdist1 =
        SphereGeometry::square_euclidean_distance(xyz, xyz_ctr1);
    const Real sqdist2 =
        SphereGeometry::square_euclidean_distance(xyz, xyz_ctr2);

    const Real h1 = hmax * std::exp(-beta * sqdist1);
    const Real h2 = hmax * std::exp(-beta * sqdist2);

    return h1 + h2;
  }

  KOKKOS_INLINE_FUNCTION
  Real operator()(const Real& x, const Real& y, const Real& z) const {
    const Real xyz[3] = {x, y, z};
    const Real xyz_ctr1[3] = {std::cos(lon1) * std::cos(lat1),
                              std::sin(lon1) * std::cos(lat1), std::sin(lat1)};
    const Real xyz_ctr2[3] = {std::cos(lon2) * std::cos(lat2),
                              std::sin(lon2) * std::cos(lat2), std::sin(lat2)};

    const Real sqdist1 =
        SphereGeometry::square_euclidean_distance(xyz, xyz_ctr1);
    const Real sqdist2 =
        SphereGeometry::square_euclidean_distance(xyz, xyz_ctr2);

    const Real h1 = hmax * std::exp(-beta * sqdist1);
    const Real h2 = hmax * std::exp(-beta * sqdist2);

    return h1 + h2;
  }
};

struct MovingVorticesTracer {
  typedef SphereGeometry geo;
  static constexpr bool IsVorticity = false;
  static constexpr Real u0 = 2 * constants::PI / 12;

  KOKKOS_INLINE_FUNCTION
  MovingVorticesTracer() = default;

  inline std::string name() const { return "MovingVorticesTracer"; }

  template <typename CVType>
  KOKKOS_INLINE_FUNCTION Real operator()(const CVType xyz,
                                         const Real t = 0) const {
    const Real lat = SphereGeometry::latitude(xyz);
    const Real lon = SphereGeometry::longitude(xyz);

    const Real lambda_prime = atan4(-std::cos(lon - u0 * t), std::tan(lat));

    const Real rho = 3 * sqrt(1 - square(cos(lat)) * square(sin(lon - u0 * t)));
    const Real omg = 1.5 * sqrt(3.0) * u0 * std::tanh(rho) *
                     FloatingPoint<Real>::safe_denominator(std::cosh(rho));

    return 1 - std::tanh(0.2 * rho * std::sin(lambda_prime - omg * t));
  }

  KOKKOS_INLINE_FUNCTION
  Real operator()(const Real& x, const Real& y, const Real& z,
                  const Real& t = 0) const {
    const Real xyz[3] = {x, y, z};
    const Real lat = SphereGeometry::latitude(xyz);
    const Real lon = SphereGeometry::longitude(xyz);

    const Real lambda_prime = atan4(-std::cos(lon - u0 * t), std::tan(lat));

    const Real rho = 3 * sqrt(1 - square(cos(lat)) * square(sin(lon - u0 * t)));
    const Real omg = 1.5 * sqrt(3.0) * u0 * std::tanh(rho) *
                     FloatingPoint<Real>::safe_denominator(std::cosh(rho));

    return 1 - std::tanh(0.2 * rho * std::sin(lambda_prime - omg * t));
  }
};

struct LatitudeTracer {
  typedef SphereGeometry geo;
  static constexpr bool IsVorticity = false;
  KOKKOS_INLINE_FUNCTION
  LatitudeTracer() = default;

  inline std::string name() const { return "initial_latitude"; }

  template <typename CVType>
  KOKKOS_INLINE_FUNCTION Real operator()(const CVType xyz) const {
    return SphereGeometry::latitude(xyz);
  }

  KOKKOS_INLINE_FUNCTION
  Real operator()(const Real& x, const Real& y, const Real& z) const {
    const Real xyz[3] = {x, y, z};
    return SphereGeometry::latitude(xyz);
  }
};

template <typename Geo>
struct FtleTracer {
  typedef Geo geo;
  static constexpr bool IsVorticity = false;

  KOKKOS_INLINE_FUNCTION
  FtleTracer() = default;

  inline std::string name() const {return "ftle";}

  template <typename PtType>
  KOKKOS_INLINE_FUNCTION
  Real operator() (const PtType& x) const {return 0;}
};

struct SphereXYZTrigTracer {
  typedef SphereGeometry geo;
  static constexpr bool IsVorticity = false;

  KOKKOS_INLINE_FUNCTION
  SphereXYZTrigTracer() = default;

  inline std::string name() const { return "SphereXYZTrigTracer"; }

  template <typename CVT>
  KOKKOS_INLINE_FUNCTION Real operator()(const CVT xyz) const {
    return 0.5 * (1 + sin(3 * xyz[0]) * sin(3 * xyz[1]) * sin(4 * xyz[2]));
  }

  template <typename CVT>
  KOKKOS_INLINE_FUNCTION Real laplacian(const CVT xyz) const {
    return -3*xyz[1]*cos(3*xyz[1])*(4*xyz[2]*cos(4*xyz[2])*sin(3*xyz[0]) + (3*xyz[0]*cos(3*xyz[0]) + sin(3*xyz[0]))*sin(4*xyz[2])) - 0.5*sin(3*xyz[1])*(6*xyz[0]*cos(3*xyz[0])*(4*xyz[2]*cos(4*xyz[2]) + sin(4*xyz[2])) +
sin(3*xyz[0])*(8*xyz[2]*cos(4*xyz[2]) + (25 - 7*square(xyz[2]))*sin(4*xyz[2])));
  }
};

}  // namespace Lpm

#endif
