#ifndef _LPM_UTILITIES_HPP
#define _LPM_UTILITIES_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include <string>

#include "Kokkos_Core.hpp"
#include <cmath>

namespace Lpm {

  /// Inverse tangent with quadrant information, but with output range in [0, 2*pi) instead of (-pi, pi]
  KOKKOS_INLINE_FUNCTION
  Real atan4(const Real y, const Real x){
    Real result = 0.0;
	if ( x == 0.0 )
	{
		if ( y > 0.0 )
			result = 0.5 * PI;
		else if ( y < 0.0 )
			result = 1.5 * PI;
		else if ( y == 0.0 )
			result = 0.0;
	}
	else if ( y == 0 )
	{
		if ( x > 0.0 )
			result = 0.0;
		else if ( x < 0.0 )
			result = PI;
	}
	else
	{
		Real theta = std::atan2( std::abs(y), std::abs(x) );
		if ( x > 0.0 && y > 0.0 )
			result = theta;
		else if ( x < 0.0 && y > 0.0 )
			result = PI - theta;
		else if ( x < 0.0 && y < 0.0 )
			result = PI + theta;
		else if ( x > 0.0 && y < 0.0 )
			result = 2.0 * PI - theta;
	}
	return result;
  }
  
  /// Determinant of a 2x2 matrix
  inline Real twoByTwoDeterminant(const Real a, const Real b, const Real c, const Real d) {return a*d - b*c;}

   /// Quadratic formula
   void quadraticRoots(Real& r1, Real& r2, const Real a, const Real b, const Real c);
   

    /// square a scalar
    KOKKOS_INLINE_FUNCTION 
    Real square(const Real& x) {return x*x;}
    /// sgn function
    KOKKOS_INLINE_FUNCTION 
    Real sign(const Real& a) {return (a>0 ? 1 : (a < 0 ? -1 : 0));}
    /// cube a scalar
    KOKKOS_INLINE_FUNCTION 
    Real cube(const Real& x) {return x*x*x;}    
   
}
#endif
