#ifndef LPM_CONSTANTS_HPP
#define LPM_CONSTANTS_HPP

#include "LpmConfig.h"

namespace Lpm {

namespace constants {

/// Pi
static constexpr Real PI = 3.1415926535897932384626433832795027975;
/// Radians to degrees conversion factor
static constexpr Real RAD2DEG = 180.0 / PI;

/// Gravitational acceleration (meters / s2)
static constexpr Real GRAVITY = 9.80616;

/// Mean sea level radius of the Earth (meters)
static constexpr Real EARTH_RADIUS_METERS = 6371220.0;

/// One sidereal day, in units of seconds (s)
static constexpr Real SIDEREAL_DAY_SEC = 24.0 * 3600.0;

/// Rotational rate of Earth about its z-axis
static constexpr Real EARTH_OMEGA_HZ = 2.0 * PI / SIDEREAL_DAY_SEC;

/// Floating point zero
static constexpr Real ZERO_TOL = 1.0e-14;

/// Null index
static constexpr Index NULL_IND = -1;

/// Lock Index
static constexpr Index LOCK_IND = -2;

/// Maximum number of edges incident to any vertex
static constexpr Int MAX_VERTEX_DEGREE = 10;

}  // namespace constants
}  // namespace Lpm

#endif
