#ifndef LPM_SURFACE_GALLERY_HPP
#define LPM_SURFACE_GALLERY_HPP

#include "LpmConfig.h"
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
};

struct ZeroBottom {
  template <typename CV>
  KOKKOS_INLINE_FUNCTION
  Real operator() (const CV x) {return 0;}

  template <typename CV>
  KOKKOS_INLINE_FUNCTION
  Real laplacian(const CV x) {return 0;}
};

}  // namespace Lpm

#endif
