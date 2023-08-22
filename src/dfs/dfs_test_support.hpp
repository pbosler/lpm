#ifndef DFS_TEST_SUPPORT_HPP
#define DFS_TEST_SUPPORT_HPP

#include "kokkos_dfs_types.hpp"

namespace SpherePoisson {
    // my test grid
    void coords(view_1d<Real> lat, view_1d<Real> lon);
    
    // my test vorticity function
    void my_vort(view_1d<Real> lat, view_1d<Real> lon, view_2d<Real> vort);

    // resulting true velocity from the above define vorticity
    void true_velocity(view_1d<Real> lat, view_1d<Real> lon, view_2d<Real> u, view_2d<Real> v, view_2d<Real> w);

    // max norm of the error
    template<typename T>
    Real max_error(view_2d<T> u_true, view_2d<T> u);

    // test function for dfs interpolation
    void test_fun(view_1d<Real> lat, view_1d<Real> lon, view_2d<Real> f);

}
#endif

