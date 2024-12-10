#ifndef LPM_KOKKOS_DEFS_HPP
#define LPM_KOKKOS_DEFS_HPP

#include "Kokkos_Core.hpp"
#include "LpmConfig.h"

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
typedef ko::DefaultExecutionSpace::memory_space DevMemory;
typedef ko::HostSpace::memory_space HostMemory;

/// Devices
typedef ko::Device<DevExe, DevMemory> Dev;
typedef ko::Device<HostExe, HostMemory> Host;

/// View to a single integer
typedef ko::View<Index, Dev> n_view_type;  // view() = n
/// View to a scalar array
typedef ko::View<Real*, Dev> scalar_view_type;
typedef ko::View<const Real*, Dev> const_scalar_view;
/// View to an index array
typedef ko::View<Index*, Dev> index_view_type;
typedef ko::View<Index* [4], Dev> quad_tree_view;
/// View to a bool array
typedef ko::View<bool*, Dev> mask_view_type;
typedef ko::View<const bool*, Dev> const_mask_view;

template <typename T>
using view_1d = Kokkos::View<T*>;

template <typename T>
using view_2d = Kokkos::View<T**>;

template <typename T>
using view_r3pts = Kokkos::View<T*[3]>;

}  // namespace Lpm

#endif
