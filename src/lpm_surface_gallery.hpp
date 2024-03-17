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

}  // namespace Lpm

#endif
