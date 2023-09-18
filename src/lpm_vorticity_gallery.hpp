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

struct SolidBodyRotation {
  typedef SphereGeometry geo;
  static constexpr Real OMEGA = 2 * constants::PI;

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

struct NitscheStricklandVortex {
  typedef PlaneGeometry geo;
  static constexpr Real b = 0.5;

  inline Real operator()(const Real& x, const Real& y, const Real& z) const {
    return 0;
  }

  KOKKOS_INLINE_FUNCTION
  Real operator()(const Real& x, const Real& y) const {
    const Real rsq = square(x) + square(y);
    const Real r = sqrt(rsq);
    return (3 * safe_divide(r) - 2 * b * r) * rsq * std::exp(-b * rsq);
  }

  inline std::string name() const { return "Nitsche&Strickland"; }

  KOKKOS_INLINE_FUNCTION
  void init_velocity(Real& u, Real& v, const Real& x, const Real& y) const {
    const Real rsq = square(x) + square(y);
    const Real utheta = rsq * std::exp(-b * rsq);
    const Real theta = std::atan2(y, x);
    u = -utheta * std::sin(theta);
    v = utheta * std::cos(theta);
  }
};

struct GaussianVortexSphere {
  typedef SphereGeometry geo;
  Real gauss_const;
  Real vortex_strength;
  Real shape_parameter;
  Kokkos::Tuple<Real, 3> xyz_ctr;

  KOKKOS_INLINE_FUNCTION
  GaussianVortexSphere(const Real str = 4 * constants::PI, const Real b = 4,
                       const Real init_lon = 0,
                       const Real init_lat = constants::PI / 20)
      : gauss_const(0),
        vortex_strength(str),
        xyz_ctr({cos(init_lon) * cos(init_lat), sin(init_lon) * cos(init_lat),
                 sin(init_lat)}) {}

  KOKKOS_INLINE_FUNCTION
  void set_gauss_const(const Real vorticity_sum, const Index n_leaf_faces) {
    gauss_const = vorticity_sum / (4 * constants::PI);
  }

  KOKKOS_INLINE_FUNCTION
  Real operator()(const Real& x, const Real& y, const Real& z) const {
    const Real distsq = 1.0 - x * xyz_ctr[0] - y * xyz_ctr[1] - z * xyz_ctr[2];
    return vortex_strength * exp(-square(shape_parameter) * distsq) -
           gauss_const;
  }

  Real operator()(const Real& x, const Real& y) const { return 0; }

  inline std::string name() const { return "SphericalGaussianVortex"; }
};

struct RossbyHaurwitz54 {
  typedef SphereGeometry geo;
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

#ifdef LPM_USE_BOOST
inline Real lamb_dipole_vorticity(const Real x, const Real y, const Real xctr,
                                  const Real yctr, const Real dipole_radius,
                                  const Real dipole_strength) {
  static constexpr Real LAMB_K0 = 3.8317;
  const Real r = sqrt(square(x - xctr) + square(y - yctr));
  Real result = 0;
  if ((r < dipole_radius) and !FloatingPoint<Real>::zero(r)) {
    const Real k = LAMB_K0 / dipole_radius;
    const Real sintheta = y / r;
    const Real denom = cyl_bessel_j(0, LAMB_K0);
    result =
        -2 * dipole_strength * k * cyl_bessel_j(1, k * r) * sintheta / denom;
  }
  return result;
}

struct CollidingDipolePairPlane {
  typedef PlaneGeometry geo;
  Real dipole_strengthA;
  Real dipole_radiusA;
  Kokkos::Tuple<Real, 2> xyz_ctrA;
  Real dipole_strengthB;
  Real dipole_radiusB;
  Kokkos::Tuple<Real, 2> xyz_ctrB;

  CollidingDipolePairPlane(const Real strA, const Real radA,
                           const Tuple<Real, 2> ctrA, const Real strB,
                           const Real radB, const Tuple<Real, 2> ctrB)
      : dipole_strengthA(strA),
        dipole_radiusA(radA),
        xyz_ctrA(ctrA),
        dipole_strengthB(strB),
        dipole_radiusB(radB),
        xyz_ctrB(ctrB) {}

  inline Real operator()(const Real& x, const Real& y, const Real& z) const {
    return 0;
  }

  inline Real operator()(const Real& x, const Real& y) const {
    return lamb_dipole_vorticity(x, y, xyz_ctrA[0], xyz_ctrA[1], dipole_radiusA,
                                 dipole_strengthA) +
           lamb_dipole_vorticity(x, y, xyz_ctrB[0], xyz_ctrB[1], dipole_radiusB,
                                 dipole_strengthB);
  }

  std::string name() const { return "PlanarCollidingDipoles"; }
};
#endif

}  // namespace Lpm
#endif
