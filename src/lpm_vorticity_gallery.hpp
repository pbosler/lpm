#ifndef LPM_VORTICITY_GALLERY_HPP
#define LPM_VORTICITY_GALLERY_HPP

#include <cmath>
#include <memory>

#include "Kokkos_Core.hpp"
#include "LpmConfig.h"
#include "lpm_constants.hpp"
#include "lpm_geometry.hpp"
#include "util/lpm_floating_point.hpp"
#include "util/lpm_math.hpp"
#include "util/lpm_tuple.hpp"

namespace Lpm {
using Kokkos::Tuple;

struct TotalVorticity {
  scalar_view_type rel_vort;
  scalar_view_type area;

  TotalVorticity(const scalar_view_type& zeta, const scalar_view_type a) :
    rel_vort(zeta),
    area(a) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index i, Real& sum) const {
    sum += rel_vort(i) * area(i);
  }
};

struct SolidBodyRotation {
  typedef SphereGeometry geo;
  static constexpr Real OMEGA = 2 * constants::PI;
  static constexpr bool IsVorticity = true;

  KOKKOS_INLINE_FUNCTION
  Real operator()(const Real& x, const Real& y, const Real& z) const {
    return 2 * OMEGA * z;
  }

  template <typename PtView> KOKKOS_INLINE_FUNCTION
  Real operator() (const PtView& pt) const {
    return 2*OMEGA*pt[2];
  }

  inline Real operator()(const Real& x, const Real& y) const { return 0; }

  inline std::string name() const { return "rotation"; }

  KOKKOS_INLINE_FUNCTION
  void init_velocity(Real& u, Real& v, Real& w, const Real& x, const Real& y,
                     const Real& z) const {
    u = -OMEGA * y;
    v = OMEGA * x;
    w = 0;
  }
};

struct GaussianVortexSphere {
  typedef SphereGeometry geo;
  static constexpr bool IsVorticity = true;
  Real gauss_const;
  Real vortex_strength;
  Real shape_parameter;
  Kokkos::Tuple<Real, 3> xyz_ctr;

  KOKKOS_INLINE_FUNCTION
  GaussianVortexSphere(const Real str = 4 * constants::PI, const Real b = 4,
                       const Real init_lon = 0,
                       const Real init_lat = constants::PI / 20)
      : gauss_const(0),
        shape_parameter(b),
        vortex_strength(str),
        xyz_ctr({cos(init_lon) * cos(init_lat), sin(init_lon) * cos(init_lat),
                 sin(init_lat)}) {  }

  KOKKOS_INLINE_FUNCTION
  void set_gauss_const(const Real vorticity_sum) {
    gauss_const = vorticity_sum / (4 * constants::PI );
  }

  KOKKOS_INLINE_FUNCTION
  Real operator()(const Real& x, const Real& y, const Real& z) const {
    const Real distsq = 1.0 - x * xyz_ctr[0] - y * xyz_ctr[1] - z * xyz_ctr[2];
    const Real zeta = vortex_strength * exp(-square(shape_parameter) * distsq) -
           gauss_const;
    return zeta;
  }

  template <typename PtType>
  KOKKOS_INLINE_FUNCTION
  Real operator() (const PtType& xyz) const {
    const Real distsq = 1.0 - xyz[0] * xyz_ctr[0] - xyz[1] * xyz_ctr[1] - xyz[2] * xyz_ctr[2];
    const Real zeta = vortex_strength * exp(-square(shape_parameter) * distsq) -
           gauss_const;
    return zeta;
  }

  Real operator()(const Real& x, const Real& y) const { return 0; }

  inline std::string name() const { return "SphericalGaussianVortex"; }
};

struct PolarVortexParams {
  Real vortex_strength; /// maximum strength of polar vortex
  Real vortex_shape_b; /// vortex shape parameter
  Real forcing_tfull; /// time of full forcing onset
  Real forcing_tend; /// end time of forcing
  Real forcing_F0; /// maximum strength of forcing

  KOKKOS_INLINE_FUNCTION
  PolarVortexParams(const Real zeta0, const Real zetab, const Real tfull, const Real tend, const Real F0) :
    vortex_strength(zeta0), vortex_shape_b(zetab),
    forcing_tfull(tfull), forcing_tend(tend), forcing_F0(F0)
    {
      LPM_KERNEL_ASSERT(zeta0 > 0);
      LPM_KERNEL_ASSERT(zetab >= 1);
      LPM_KERNEL_ASSERT(tfull >= 0);
      LPM_KERNEL_ASSERT(tend > tfull);
      LPM_KERNEL_ASSERT(F0 >= 0);
    }
};

struct JM86PolarVortex {
  typedef SphereGeometry geo;
  static constexpr bool IsVorticity = true;
  Real gauss_const;
  Real strength;
  Real b;
  static constexpr Real theta0 = 0.5*constants::PI;

  KOKKOS_INLINE_FUNCTION
  JM86PolarVortex(const JM86PolarVortex& other) = default;

  KOKKOS_INLINE_FUNCTION
  JM86PolarVortex(const PolarVortexParams& params) :
    gauss_const(0),
    strength(params.vortex_strength),
    b(params.vortex_shape_b) {}

  KOKKOS_INLINE_FUNCTION
  JM86PolarVortex(const Real str = 2*constants::PI, const Real b = 2.0) :
    gauss_const(0),
    strength(str),
    b(b) {}

  KOKKOS_INLINE_FUNCTION
  void set_gauss_const(const Real vorticity_sum) {
    gauss_const = vorticity_sum / (4 * constants::PI );
  }

  KOKKOS_INLINE_FUNCTION
  Real operator()(const Real& x, const Real& y, const Real& z) const {
    const Real xyz[3] = {x,y,z};
//     const Real theta = SphereGeometry::latitude(xyz);
//     const Real exp_arg = -2*square(b)*(1-cos(theta0)*cos(theta)-sin(theta0)*sin(theta));
//     const Real exp_mul = cos(theta)*(2*square(b)*(cos(theta0)*sin(theta)-sin(theta0)*cos(theta)) + sin(theta));
//     const Real zeta = strength*exp_mul*exp(exp_arg) - gauss_const;
//     return zeta;
      return strength * exp(-2*square(b)*(1-z));
  }

  template <typename PtType>
  KOKKOS_INLINE_FUNCTION
  Real operator() (const PtType& xyz) const {
    const Real zeta = this->operator()(xyz[0], xyz[1],xyz[2]);
    return zeta;
  }

  Real operator()(const Real& x, const Real& y) const { return 0; }

  inline std::string name() const { return "JM86PolarVortex"; }
};


struct JM86Forcing {
  typedef SphereGeometry geo;
  static constexpr Real b0 = 0.5*constants::PI;
  static constexpr Real b1 = 0.25*constants::PI;
  static constexpr Real b2 = constants::PI/3;
  static constexpr Real b3 = 8;
  Real tfull;
  Real tend;
  Real tstar;
  Real F0;

  KOKKOS_INLINE_FUNCTION
  JM86Forcing(const JM86Forcing& other) = default;

  KOKKOS_INLINE_FUNCTION
  JM86Forcing(const PolarVortexParams& params) :
    tfull(params.forcing_tfull),
    tend(params.forcing_tend),
    tstar(params.forcing_tfull-params.forcing_tend),
    F0(params.forcing_F0) {}

  KOKKOS_INLINE_FUNCTION
  JM86Forcing(const Real tfull = 4, const Real tend = 15,
    const Real F0 = 6*constants::PI/5) :
    tfull(tfull), tend(tend), tstar(tend-tfull),
    F0(F0) {LPM_KERNEL_ASSERT(tend > tfull);}

  KOKKOS_INLINE_FUNCTION
  Real forcing_a(const Real t) const {
    Real result = 0.0;
    if (t < tfull) {
      result = square(sin(constants::PI * t / (2*tfull)));
    }
    else {
      if (tfull <= t and t < tstar) {
        result = 1.0;
      }
      else {
        if (tstar <= t and t < tend) {
          result = square(sin(constants::PI * (tend - t)/(2*tfull)));
        }
      }
    }
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  Real forcing_aprime(const Real t) const {
    Real result = 0.0;
    if (t < tfull) {
      result = constants::PI * sin(constants::PI*t/tfull)/(2*tfull);
    }
    else {
      if (tstar <= t and t < tend) {
        const Real sinarg = constants::PI * (t - tstar)/tfull;
        result = -constants::PI*sin(sinarg)/(2*tfull);
      }
    }
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  Real forcing_b(const Real theta) const {
    return -2*constants::PI*(theta-b0)*(theta-b1)*exp(-b3*square(theta - b2));
  };

  KOKKOS_INLINE_FUNCTION
  Real forcing_bprime(const Real theta) const {
    const Real polyfac = 2*constants::PI * (2*b3*(theta-b0)*(theta-b1)*(theta-b2) - (theta-b1) - (theta-b0));
    return polyfac * exp(-b3*square(theta-b2));
  }


  template <typename XYZType> KOKKOS_INLINE_FUNCTION
  Real operator() (const XYZType& xyz, const Real t) const {
    const Real theta = SphereGeometry::latitude(xyz);
    const Real lambda = SphereGeometry::longitude(xyz);
    return F0 * forcing_a(t) * forcing_b(theta) * cos(lambda);
  }

  template <typename XyzType, typename VelType> KOKKOS_INLINE_FUNCTION
  Real derivative(const XyzType& xyz, const VelType& uvw, const Real t) const {
    const Real theta = SphereGeometry::latitude(xyz);
    const Real lambda = SphereGeometry::longitude(xyz);
    const Real uzonal = -sin(lambda)*uvw[0] + cos(lambda)*uvw[1];
    const Real vmerid = -sin(theta)*cos(lambda)*uvw[0] - sin(theta)*sin(lambda)*uvw[1] + cos(theta)*uvw[2];
    const Real partial_t = F0*forcing_aprime(t)*forcing_b(theta)*cos(lambda);
    const Real udotgrad = F0*forcing_a(t) * (vmerid*forcing_bprime(theta)*cos(lambda) -
      uzonal*forcing_b(theta)*sin(lambda)*FloatingPoint<Real>::safe_denominator(cos(theta)));
    return partial_t + udotgrad;
  }
};


struct RossbyHaurwitz54 {
  typedef SphereGeometry geo;
  static constexpr bool IsVorticity = true;
  Real u0;
  Real rh54_amplitude;

  KOKKOS_INLINE_FUNCTION
  RossbyHaurwitz54(const Real zonal_background_velocity = 0,
                   const Real wave_amp = 1)
      : u0(zonal_background_velocity), rh54_amplitude(wave_amp) {}

  KOKKOS_INLINE_FUNCTION
  Real operator()(const Real& x, const Real& y) const { return 0; }

  std::string name() const { return "RossbyHaurwitz54"; }

  KOKKOS_INLINE_FUNCTION
  void set_stationary_wave_speed(const Real& Omega=2*constants::PI) {
    u0 = Omega / 14;
  }

  KOKKOS_INLINE_FUNCTION
  Real legendreP54(const Real z) const { return z * square(square(z) - 1); }

  KOKKOS_INLINE_FUNCTION
  Real operator()(const Real& x, const Real& y, const Real& z) const {
    const Real lon = atan4(y, x);
    return 2 * u0 * z + 30 * rh54_amplitude * cos(4 * lon) * legendreP54(z);
  }

  template <typename PtType>
  KOKKOS_INLINE_FUNCTION
  Real operator() (const PtType& xyz) const {
    return 2*u0*xyz[2] + 30 * rh54_amplitude * cos(4*SphereGeometry::longitude(xyz)) * legendreP54(xyz[2]);
  }

  template <typename PtType>
  KOKKOS_INLINE_FUNCTION
  Real laplacian(const PtType& xyz) const {
    return -4*u0*xyz[2] - 900 * rh54_amplitude * cos(4*SphereGeometry::longitude(xyz)) * legendreP54(xyz[2]);
  }
};


//
struct RossbyHaurwitzR {
  typedef SphereGeometry geo;
  Real u0;
  Real rh54_amplitude;
  Int R;

  KOKKOS_INLINE_FUNCTION
  RossbyHaurwitzR(const Real zonal_background_velocity = 0,
                   const Real wave_amp = 1, Int R_=6)
      : u0(zonal_background_velocity), rh54_amplitude(wave_amp), R (R_) {}

  KOKKOS_INLINE_FUNCTION
  Real operator()(const Real& x, const Real& y) const { return 0; }

  std::string name() const { return "RossbyHaurwitzR"; }

  KOKKOS_INLINE_FUNCTION
  Real legendrePR(const Real z) const { return z * pow(1-square(z), R/2); }

  KOKKOS_INLINE_FUNCTION
  Real operator()(const Real& x, const Real& y, const Real& z) const {
    const Real lon = atan4(y, x);
    return 2 * u0 * z + (R+1)*(R+2) * rh54_amplitude * cos(R * lon) * legendrePR(z);
  }
};
//

//#ifdef LPM_USE_BOOST
//inline Real lamb_dipole_vorticity(const Real x, const Real y, const Real xctr,
/** Returns the vorticity of a compactly supported Lamb dipole.

  @param [in] x x-coordinate where vorticity will be evaluated
  @param [in] y y-coordinate where vorticity will be evaluated
  @param [in] xctr x-coordinate of dipole center
  @param [in] yctr y-coordinate of dipole center
  @param [in] dipole_radius radius of support (vorticity is zero outside of this radius)
  @param [in] dipole_strength strength of dipole
  @return vorticity
*/
KOKKOS_INLINE_FUNCTION
Real lamb_dipole_vorticity(const Real x, const Real y, const Real xctr,
                                  const Real yctr, const Real dipole_radius,
                                  const Real dipole_strength) {
  static constexpr Real LAMB_K0 = 3.8317;
  const Real r = sqrt(square(x - xctr) + square(y - yctr));
  Real result = 0;
  if ((r < dipole_radius) and !FloatingPoint<Real>::zero(r)) {
    const Real k = LAMB_K0 / dipole_radius;
    const Real sintheta = y / r;
    const Real denom = bessel_j0(LAMB_K0);
    result =
        -2 * dipole_strength * k * bessel_j1(k * r) * sintheta / denom;
  }
  return result;
}

struct CollidingDipolePairPlane {
  typedef PlaneGeometry geo;
  static constexpr bool IsVorticity = true;
  Real dipole_strengthA;
  Real dipole_radiusA;
  Kokkos::Tuple<Real, 2> xyz_ctrA;
  Real dipole_strengthB;
  Real dipole_radiusB;
  Kokkos::Tuple<Real, 2> xyz_ctrB;

  KOKKOS_INLINE_FUNCTION
  CollidingDipolePairPlane()
      : dipole_strengthA(1),
        dipole_radiusA(1),
        xyz_ctrA({-1.5,0}),
        dipole_strengthB(-1),
        dipole_radiusB(1),
        xyz_ctrB({ 1.5,0}) {}

  KOKKOS_INLINE_FUNCTION
  CollidingDipolePairPlane(const Real strA, const Real rA, const Kokkos::Tuple<Real,2> ctrA,
    const Real strB, const Real rB, const Kokkos::Tuple<Real,2> ctrB) :
    dipole_strengthA(strA),
    dipole_radiusA(rA),
    xyz_ctrA(ctrA),
    dipole_strengthB(strB),
    dipole_radiusB(rB),
    xyz_ctrB(ctrB) {}

  inline Real operator() (const Real& x, const Real& y, const Real& z) const {
    return 0;
  }

  KOKKOS_INLINE_FUNCTION
  Real operator()(const Real& x, const Real& y) const {
    return lamb_dipole_vorticity(x, y, xyz_ctrA[0], xyz_ctrA[1], dipole_radiusA,
                                 dipole_strengthA) +
           lamb_dipole_vorticity(x, y, xyz_ctrB[0], xyz_ctrB[1], dipole_radiusB,
                                 dipole_strengthB);
  }

  template <typename CV> KOKKOS_INLINE_FUNCTION
  Real operator() (const CV& xy) const {
    return lamb_dipole_vorticity(xy[0], xy[1], xyz_ctrA[0], xyz_ctrA[1], dipole_radiusA,
              dipole_strengthA) +
           lamb_dipole_vorticity(xy[0], xy[1], xyz_ctrB[0], xyz_ctrB[1], dipole_radiusB,
              dipole_strengthB);
  }


  std::string name() const { return "PlanarCollidingDipoles"; }
};

/** Relative vorticity from shallow water test case 2 in
  Williamson et al., 1992.
*/
struct SphereTestCase2Vorticity {
  static constexpr Real sphere_radius = 1.0;
  static constexpr Real u0 = 2*constants::PI / 12;
  static constexpr bool IsVorticity = true;

  KOKKOS_INLINE_FUNCTION
  SphereTestCase2Vorticity() = default;

  template <typename CV> KOKKOS_INLINE_FUNCTION
  Real operator() (const CV& xyz) const {
    return 2*u0*xyz[2];
  }

  std::string name() const {return "SphereTestCase2Vorticity";}
};

}  // namespace Lpm
#endif
