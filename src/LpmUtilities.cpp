#include "LpmUtilities.h"
#include <cmath>
#include <limits>

namespace Lpm {

  scalar_type atan4 (const scalar_type y, const scalar_type x) {
    scalar_type result = 0.0;
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
		scalar_type theta = std::atan2( std::abs(y), std::abs(x) );
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
  
  void quadraticRoots(scalar_type& r1, scalar_type& r2, const scalar_type a, const scalar_type b, const scalar_type c) {
     const scalar_type apa = 2.0 * a;
     scalar_type disc = b*b - 4.0*a*c;
     if (std::abs(disc) < ZERO_TOL) {
        disc = 0.0;
     }
     else if (disc < -ZERO_TOL) {
        r1 = std::numeric_limits<scalar_type>::max();
        r2 = std::numeric_limits<scalar_type>::max();
     }
     else {
        const scalar_type rdisc = std::sqrt(disc);
        r1 = (-b + rdisc)/apa;
        r2 = (-b - rdisc)/apa;
     }
  }


}
