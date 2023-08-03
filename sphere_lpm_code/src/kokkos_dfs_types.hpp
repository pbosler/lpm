#ifndef KOKKOS_DFS_TYPES_HPP
#define KOKKOS_DFS_TYPES_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

namespace SpherePoisson {

typedef double Real;
typedef int Int;
//typedef Kokkos::complex<double> Complex;
typedef std::complex<double> Complex;

template <typename T>
using view_1d = Kokkos::View<T*>;

template <typename T>
using view_2d = Kokkos::View<T**>;

template <typename T>
using view_r3pts = Kokkos::View<T*[3]>;

} // namespace SpherePoisson
#endif
