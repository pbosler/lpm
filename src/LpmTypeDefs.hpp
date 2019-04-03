#ifndef LPM_TYPEDEFS_HPP
#define LPM_TYPEDEFS_HPP

#include "LpmConfig.h"
#include "Kokkos_Core.hpp"

namespace Lpm {

/// Real number type
typedef double Real;
/// Integer type
typedef int Int;
/// Memory index type
typedef int Index;

typedef Kokkos::View<Real*> RealArray;
typedef Kokkos::View<Index*> IndexArray;


/// Pi
static const Real PI = 3.1415926535897932384626433832795027975;
/// Radians to degrees conversion factor
static const Real RAD2DEG = 180.0 / PI;

/// Gravitational acceleration
static const Real G = 9.80616;

/// Mean sea level radius of the Earth (meters)
static const Real EARTH_RADIUS_METERS = 6371220.0;

/// One sidereal day, in units of seconds
static const Real SIDEREAL_DAY_SEC = 24.0 * 3600.0;

/// Rotational rate of Earth about its z-axis
static const Real EARTH_OMEGA_HZ = 2.0 * PI / SIDEREAL_DAY_SEC;

/// Floating point zero
static const Real ZERO_TOL = 1.0e-13;

/// Supported geometries
enum GeometryType {PLANAR_GEOMETRY, SPHERICAL_SURFACE_GEOMETRY};

}
#endif
