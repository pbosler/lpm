#ifndef _LPM_UTILITIES_HPP
#define _LPM_UTILITIES_HPP

#include "LpmConfig.h"
#include "LpmTypeDefs.hpp"
#include <string>

#include "Kokkos_Core.hpp"
#include <cmath>
#include <algorithm>

namespace Lpm {

  /// Inverse tangent with quadrant information, but with output range in [0, 2*pi) instead of (-pi, pi]
  Real atan4(const Real y, const Real x);
  
  /// Determinant of a 2x2 matrix
  inline Real twoByTwoDeterminant(const Real a, const Real b, const Real c, const Real d) {return a*d - b*c;}

   /// Quadratic formula
   void quadraticRoots(Real& r1, Real& r2, const Real a, const Real b, const Real c);
   
#ifdef HAVE_KOKKOS
    /// square a scalar
    KOKKOS_INLINE_FUNCTION Real square(const Real& x) {return x*x;}
    /// sgn function
    KOKKOS_INLINE_FUNCTION Real sign(const Real& a) {return (a>0 ? 1 : (a < 0 ? -1 : 0));}
    /// cube a scalar
    KOKKOS_INLINE_FUNCTION Real cube(const Real& x) {return x*x*x;}    
#else
    /// square a scalar
    inline Real square(const Real& x) {return x*x;}
    /// sgn function
    inline Real sign(const Real& a) {return (a>0 ? 1 : (a<0 ? -1 : 0));}
    /// cube a scalar
    inline Real cube(const Real& x) {return x*x*x;}
#endif
#ifdef KOKKOS_ENABLE_CUDA
    template <typename T> KOKKOS_INLINE_FUNCTION 
    const T& min (const T& a, const T& b) {return a < b ? a : b;}
    
    template <typename T> KOKKOS_INLINE_FUNCTION
    const T& max (const T& a, const T& b) {return a > b ? a : b;}
    
    template <typename T> KOKKOS_INLINE_FUNCTION
    const T* max_element(const T* const begin, const T* const end) {
        const T* me = begin;
        for (const T* it=begin +1; it < end; ++it) {
            if (!(*it < *me)) me = it;
        }
        return me;
    }
    
#else
    using std::min;
    using std::max;
    using std::max_element;
#endif

   
}
#endif
