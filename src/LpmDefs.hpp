#ifndef LPM_TYPEDEFS_HPP
#define LPM_TYPEDEFS_HPP

#include <cmath>
#include <cassert>
#include <iostream>
#include <sstream>
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <algorithm>
#include "LpmConfig.h"
#include "Kokkos_Core.hpp"

namespace Lpm {
namespace ko = Kokkos;

template<typename T>
static void prarr (const std::string& name, const T* const v, const size_t n) {
  std::cerr << name << ": ";
  for (size_t i = 0; i < n; ++i) std::cerr << " " << v[i];
  std::cerr << "\n";
}

/// Error handling
#define LPM_THROW_IF(condition, message) do { \
    if (condition) {                                                    \
      std::stringstream _ss_;                                           \
      _ss_ << __FILE__ << ":" << __LINE__ << ": The condition:\n" << #condition \
        "\nled to the exception\n" << message << "\n";                  \
      throw std::logic_error(_ss_.str());                               \
    }                                                                   \
} while (0)

KOKKOS_INLINE_FUNCTION static void error (const char* const msg)
{ ko::abort(msg); }

KOKKOS_INLINE_FUNCTION static void message (const char* const msg)
{ printf("%s\n", msg); }

/// Real number type
typedef double Real;
/// Integer type
typedef int Int;
/// Memory index type
typedef int Index;
/// Memory layout
#ifdef HAVE_CUDA
typedef ko::LayoutLeft Layout;
#else
typedef ko::LayoutRight Layout;
#endif

/// Execution spaces
typedef ko::DefaultExecutionSpace DevExe;
typedef ko::HostSpace::execution_space HostExe;

/// Memory spaces
typedef ko::DefaultExecutionSpace::memory_space DevMem;
typedef ko::HostSpace HostMem;

/// Devices
typedef ko::Device<DevExe, DevMem> Dev;
typedef ko::Device<HostExe, HostMem> Host;

#ifdef HAVE_CUDA
/// 1d slice of an array
template <typename VT> KOKKOS_FORCEINLINE_FUNCTION
ko::View<typename VT::value_type*, ko::LayoutStride, typename VT::device_type, ko::MemoryTraits<ko::Unmanaged>>
slice(const VT& v, Int i) {return ko::subview(v, i, ko::ALL());}
/// explicitly const 1d slice of an array
template <typename VT> KOKKOS_FORCEINLINE_FUNCTION
ko::View<typename VT::const_value_type*, ko::LayoutStride, typename VT::device_type, ko::MemoryTraits<ko::Unmanaged>>
const_slice(const VT& v, Int i) {return ko::subview(v, i, ko::ALL());}
#else
template <typename VT> KOKKOS_FORCEINLINE_FUNCTION
typename VT::value_type*
slice(const VT& v, Int i) {return v.data() + v.extent(1)*i;}

template <typename VT> KOKKOS_FORCEINLINE_FUNCTION
typename VT::const_value_type*
const_slice(const VT& v, Int i) {return v.data() + v.extent(1)*i;}
#endif

/// Pi
static constexpr Real PI = 3.1415926535897932384626433832795027975;
/// Radians to degrees conversion factor
static constexpr Real RAD2DEG = 180.0 / PI;

/// Gravitational acceleration
static constexpr Real G = 9.80616;

/// Mean sea level radius of the Earth (meters)
static constexpr Real EARTH_RADIUS_METERS = 6371220.0;

/// One sidereal day, in units of seconds
static constexpr Real SIDEREAL_DAY_SEC = 24.0 * 3600.0;

/// Rotational rate of Earth about its z-axis
static constexpr Real EARTH_OMEGA_HZ = 2.0 * PI / SIDEREAL_DAY_SEC;

/// Floating point zero
static constexpr Real ZERO_TOL = 1.0e-14;

/// Null index
static constexpr Index NULL_IND = -1;

/// Lock Index
static constexpr Index LOCK_IND = -2;

/// View to a single integer
typedef ko::View<Index,Dev> n_view_type; // view(0) = n
/// View to a scalar array
typedef ko::View<Real*,Dev> scalar_view_type;
/// View to an index array
typedef ko::View<Index*,Dev> index_view_type;
/// View to a bool array
typedef ko::View<bool*,Dev> mask_view_type;
typedef ko::View<const bool*,Dev> const_mask_view_type;

#ifdef KOKKOS_ENABLE_CUDA
    /// GPU-friendly replacements for stdlib functions
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

/// Timers 
static timeval tic () {
  timeval t;
  gettimeofday(&t, 0);
  return t;
}
static double calc_et (const timeval& t1, const timeval& t2) {
  static const double us = 1.0e6;
  return (t2.tv_sec * us + t2.tv_usec - t1.tv_sec * us - t1.tv_usec) / us;
}
static double toc (const timeval& t1) {
  Kokkos::fence();
  timeval t;
  gettimeofday(&t, 0);
  return calc_et(t1, t);
}
static double get_memusage () {
  static const double scale = 1.0 / (1 << 10); // Memory in MB.
  rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  return ru.ru_maxrss*scale;
}
static void print_times (const std::string& name, const double* const parts,
                         const int nparts) {
  double total = 0; for (int i = 0; i < nparts; ++i) total += parts[i];
  printf("%20s %1.3e s %7.1f MB", name.c_str(), total, get_memusage());
  for (int i = 0; i < nparts; ++i) printf(" %1.3e s", parts[i]);
  printf("\n");
}
static void print_times (const std::string& name, const double total) {
   printf("%20s %1.3e s %5.1f MB\n", name.c_str(), total, get_memusage());
}


}
#endif
