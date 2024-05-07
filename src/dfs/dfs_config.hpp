#ifndef DFS_CONFIG_HPP
#define DFS_CONFIG_HPP

#include "LpmConfig.h"
#include "lpm_kokkos_defs.hpp"

namespace SpherePoisson {

using Lpm::Int;
using Lpm::Real;
using Lpm::Complex;
template <typename T>
using view_1d = Lpm::view_1d<T>;
template <typename T>
using view_2d = Lpm::view_2d<T>;
template <typename T>
using view_r3pts = Lpm::view_r3pts<T>;
template <typename T> 
using view_3d = Lpm::view_3d<T>;

} // namespace SpherePoisson

#endif
