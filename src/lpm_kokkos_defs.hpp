#ifndef LPM_KOKKOS_DEFS_HPP
#define LPM_KOKKOS_DEFS_HPP

#include "LpmConfig.h"
#include "Kokkos_Core.hpp"

namespace Lpm {
namespace ko = Kokkos;

/// Memory layout
#ifdef LPM_HAVE_CUDA
typedef ko::LayoutLeft Layout;
#else
typedef ko::LayoutRight Layout;
#endif

/// Execution spaces
typedef ko::DefaultExecutionSpace DevExe;
typedef ko::HostSpace::execution_space HostExe;
typedef typename ko::TeamPolicy<>::member_type member_type;

/// Memory spaces
typedef ko::DefaultExecutionSpace::memory_space DevMem;
typedef ko::HostSpace::memory_space HostMem;

/// Devices
typedef ko::Device<DevExe, DevMem> Dev;
typedef ko::Device<HostExe, HostMem> Host;

/// View to a single integer
typedef ko::View<Index,Dev> n_view_type; // view() = n
/// View to a scalar array
typedef ko::View<Real*,Dev> scalar_view_type;
/// View to an index array
typedef ko::View<Index*,Dev> index_view_type;
typedef ko::View<Index*[4],Dev> quad_tree_view;
/// View to a bool array
typedef ko::View<bool*,Dev> mask_view_type;
typedef ko::View<const bool*,Dev> const_mask_view_type;


/// Array slices
#ifdef LPM_HAVE_CUDA
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


} // namespace Lpm

#endif
