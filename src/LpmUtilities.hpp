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

std::string indentString(const int tab_lev);

std::string weightName(const int ndim);

/// Determinant of a 2x2 matrix
inline Real twoByTwoDeterminant(const Real a, const Real b, const Real c, const Real d) {return a*d - b*c;}

/// Quadratic formula
void quadraticRoots(Real& r1, Real& r2, const Real a, const Real b, const Real c);

/// floating point equivalence
template <typename T> KOKKOS_INLINE_FUNCTION
bool fp_equiv(const T& a, const T& b, const T& tol=ZERO_TOL) {
  return std::abs(a-b) <= tol;
}

/// square a scalar
template <typename T=Real> KOKKOS_INLINE_FUNCTION
T square(const T& x) {return x*x;}
/// sgn function
template <typename T=Real> KOKKOS_INLINE_FUNCTION
T sign(const T& a) {return (a>0 ? 1 : (a < 0 ? -1 : 0));}
/// cube a scalar
template <typename T=Real> KOKKOS_INLINE_FUNCTION
T cube(const T& x) {return x*x*x;}

/** safely divide by a real number

  @f$ \frac{1}{x} = \lim_{\epsilon \to 0} \frac{x}{x^2 + \epsilon^2} @f$

  @param x desired denominator
  @param eps regularization parameter
  @return @f$ \frac{x}{x^2 + \epsilon^2} @f$
*/
template <typename T=Real> KOKKOS_INLINE_FUNCTION
T safe_divide(const T& x, const T& eps=ZERO_TOL) {return x/(square(x) + square(eps));}

template <typename MaskViewType>
Index mask_count(const MaskViewType& mv) {
  Index result;
  ko::parallel_reduce("mask_count", mv.extent(0),
    KOKKOS_LAMBDA (const Index& i, Index& s) {
      if (mv(i)) ++s;
    }, result);
  return result;
}

std::string& tolower(std::string& s);

std::string format_strings_as_list(const char** strings, const Short n);

class ProgressBar {
  std::string name_;
  Int niter_;
  Real freq_;
  Int it_;
  Real next_;
  std::ostream& os_;

  public:
    ProgressBar(const std::string& name, const Int niterations,
      const Real write_freq = 10.0, std::ostream& os = std::cout);

    void update();
};

}
#endif
