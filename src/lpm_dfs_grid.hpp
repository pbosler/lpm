#ifndef LPM_HEADER_HPP
#define LPM_HEADER_HPP

#include "LpmConfig.h"
#include "lpm_constants.hpp"

namespace Lpm {

struct DFSGrid {

  Int nlon;
  Int nlat;

  KOKKOS_INLINE_FUNCTION
  explicit DFSGrid(const Int nl) : nlon(nl), nlat(nl/2 + 1);

  KOKKOS_INLINE_FUNCTION
  Real colatitude(const Int i) const {
    return constants::PI * i / nlat;
  }

  KOKKOS_INLINE_FUNCTION
  Real longitude(const Int j) const {
    return 2*constants::PI * j / nlon;
  }

};


} // namespace Lpm

#endif
