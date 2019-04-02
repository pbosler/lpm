#ifndef _LPM_UTILITIES_H
#define _LPM_UTILITIES_H

#include "LpmConfig.h"
#include "LpmTypeDefs.h"
#include <string>

#ifdef HAVE_KOKKOS
#include "Kokkos_Core.hpp"
#endif

#ifndef KOKKOS_ENABLE_CUDA
#include <cmath>
#include <algorithm>
#endif

namespace Lpm {

  /// Inverse tangent with quadrant information, but with output range in [0, 2*pi) instead of (-pi, pi]
  scalar_type atan4(const scalar_type y, const scalar_type x);
  
  /// Returns "length" for 1d, "area" for 2d, "volume" for 3d
  std::string weight_name(const int ndim);
  
  /// Determinant of a 2x2 matrix
  inline scalar_type twoByTwoDeterminant(const scalar_type a, const scalar_type b, const scalar_type c, const scalar_type d) {
    return a*d - b*c;}

   /// Quadratic formula
   void quadraticRoots(scalar_type& r1, scalar_type& r2, const scalar_type a, const scalar_type b, const scalar_type c);
   
   /// square a scalar
#ifdef HAVE_KOKKOS
    KOKKOS_INLINE_FUNCTION scalar_type square(const scalar_type& x) {return x*x;}
    KOKKOS_INLINE_FUNCTION scalar_type sign(const scalar_type& a) {return (a>0 ? 1 : (a < 0 ? -1 : 0));}
    KOKKOS_INLINE_FUNCTION scalar_type cube(const scalar_type& x) {return x*x*x;}    
#else
    inline scalar_type square(const scalar_type& x) {return x*x;}
    inline scalar_type sign(const scalar_type& a) {return (a>0 ? 1 : (a<0 ? -1 : 0));}
    inline scalar_type cube(const scalar_type& x) {return x*x*x;}
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
