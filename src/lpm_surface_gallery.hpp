#ifndef LPM_SURFACE_GALLERY_HPP
#define LPM_SURFACE_GALLERY_HPP

#include "LpmConfig.h"
#include "lpm_constants.hpp"
#include "lpm_geometry.hpp"

namespace Lpm {

/** @brief Returns the height (above a reference z_0 = 0) of bottom topography
  for a planar shallow water problem.
*/
struct PlanarParabolicBasin {
  typedef PlaneGeometry geo;
  static constexpr Int ndim = 2;
  static constexpr Real H = 1;
  static constexpr Real D0 = 2;
  static constexpr Real L = 3;

  KOKKOS_INLINE_FUNCTION
  PlanarParabolicBasin() = default;

  template <typename CV>
  KOKKOS_INLINE_FUNCTION Real operator()(const CV xy) const {
    return -H - D0(1 - square(xy[0] / L) - square(xy[1] / L));
  }

  template <typename CV>
  KOKKOS_INLINE_FUNCTION Real laplacian(const CV xy) const {
    return 4 * D0 / square(L);
  }

  std::string name() const {return "PlanarParabolicBasin";}
};


/** @brief Returns the height (above a reference z_0 = 0) of bottom topography
  for a planar shallow water problem.

  This function defines a Gaussian mountain centered at the origin.
*/
struct PlanarGaussianMountain {
  typedef PlaneGeometry geo;
  static constexpr Int ndim = 2;

  // Modifiable parameters
  static constexpr Real mtn_height = 0.8;  // height of underwater mountain
  static constexpr Real b=5.0;  // Gaussian shape parameter

  template <typename CV>
  KOKKOS_INLINE_FUNCTION Real operator() (const CV xy) const {
    return mtn_height * exp(-b * PlaneGeometry::norm2(xy) );
  }

  template <typename CV>
  KOKKOS_INLINE_FUNCTION Real laplacian(const CV xy) const {
    return 4 * b * mtn_height *
      (b*PlaneGeometry::norm2(xy) - 1) *
      exp(-b * PlaneGeometry::norm2(xy)) ;
  }

  std::string name() const {return "PlanarGaussianMountain";}
};

/** @brief Returns the height (above a reference z_0 = 0) of bottom topography
  for a planar shallow water problem.

  This function defines a Gaussian mountain centered at the origin.
*/
struct PlanarGaussianSurfacePerturbation {
  typedef PlaneGeometry geo;
  static constexpr Int ndim = 2;

  // Modifiable parameters
  static constexpr Real H0 = 1.0; // height of unperturbed surface
  static constexpr Real ptb_height = 0.1;  // height of surface perturbation
  static constexpr Real ptb_bx = 20;  // Gaussian shape parameter, x-direction
  static constexpr Real ptb_by = 5; // Gaussian shape parameter, y-direction
  static constexpr Real ptb_x0 = -1.125; // Center of perturbation, x-coordinate
  static constexpr Real ptb_y0 = 0; // Center of perturbation, y-coordinate

  template <typename CV>
  KOKKOS_INLINE_FUNCTION Real operator() (const CV xy) const {
    return H0 + ptb_height * exp(-(ptb_bx * square(xy(0) - ptb_x0) + ptb_by * square(xy(1) - ptb_y0)));
  }

  std::string name() const {return "PlanarGaussianSurfacePerturbation";}
};

/** Topography functor for a flat bottom
*/
struct ZeroFunctor {
  template <typename CV>
  KOKKOS_INLINE_FUNCTION
  Real operator() (const CV x) const {return 0;}

  template <typename CV>
  KOKKOS_INLINE_FUNCTION
  Real laplacian(const CV x) const {return 0;}

  std::string name() const {return "ZeroFunctor";}
};

/** Surface functor for uniform depth
*/
struct UniformDepthSurface {
  Real H0;

  KOKKOS_INLINE_FUNCTION
  explicit UniformDepthSurface(const Real h0=1) : H0(h0) {}

  template <typename CV>
  KOKKOS_INLINE_FUNCTION Real operator() (const CV x) const {return H0;}
};

struct SphereTestCase2InitialSurface {
  static constexpr Real h0 = 10;
  static constexpr Real g = 1.0;
  static constexpr Real Omega = 2*constants::PI;
  static constexpr Real u0 = 2*constants::PI / 12;


  template <typename CV>
  KOKKOS_INLINE_FUNCTION
  Real operator() (const CV xyz) const {
    const Real cos_theta_sq = 1-square(xyz[2]);
    const Real result = h0 + Omega * u0 * cos_theta_sq / g;
    return result;
  }

  std::string name() const {return "SphereTestCase2InitialSurface";}
};

struct SphereTestCase5Bottom {
  static constexpr Real h_mtn = 1E-4;
  static constexpr Real r_mtn = constants::PI/9;
  static constexpr Real r_mtn2 = r_mtn*r_mtn;
  static constexpr Real lambda_ctr = 1.5*constants::PI;
  static constexpr Real theta_ctr = constants::PI/6;

  template <typename CV>
  KOKKOS_INLINE_FUNCTION
  Real operator() (const CV xyz) const {
    const Real lambda = SphereGeometry::longitude(xyz);
    const Real theta = SphereGeometry::latitude(xyz);
    const Real ll_dist_sq = square(lambda - lambda_ctr) + square(theta - theta_ctr);
    const Real r = (r_mtn2 < ll_dist_sq ? r_mtn : sqrt(ll_dist_sq));
    return h_mtn * (1 - r / r_mtn);
  }

  std::string name() const {return "SphereTestCase5Bottom";}
};

}  // namespace Lpm

#endif
