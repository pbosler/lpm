#ifndef LPM_SURFACE_GALLERY_HPP
#define LPM_SURFACE_GALLERY_HPP

#include "LpmConfig.h"

namespace Lpm {

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

}  // namespace Lpm

#endif
