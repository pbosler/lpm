#include "LpmUtilities.hpp"
#include <cmath>
#include <limits>

namespace Lpm {

  Real atan4 (const Real y, const Real x) {
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
  
  void quadraticRoots(Real& r1, Real& r2, const Real a, const Real b, const Real c) {
     const Real apa = 2.0 * a;
     Real disc = b*b - 4.0*a*c;
     if (std::abs(disc) < ZERO_TOL) {
        disc = 0.0;
     }
     else if (disc < -ZERO_TOL) {
        r1 = std::numeric_limits<Real>::max();
        r2 = std::numeric_limits<Real>::max();
     }
     else {
        const Real rdisc = std::sqrt(disc);
        r1 = (-b + rdisc)/apa;
        r2 = (-b - rdisc)/apa;
     }
  }


}
