#ifndef DFS_INTERPOLATION_HPP
#define DFS_INTERPOLATION_HPP

#include "dfs_config.hpp"
#include <finufft.h>
#include <iostream>
#include <complex>
#include "dfs_test_support.hpp"
#include "dfs_rhs_new.hpp"
#include "dfs_doubling.hpp"


namespace SpherePoisson {

    // defines an interpolation on the sphere that exploits the
    // double Fourier sphere method and the non uniform fast
    // fourier for a fast and exponentially convergent
    // interpolation
    /*
    INPUTS:
        Evaluation points:
        th - nonuniform spaced points in the colatitude direction
        lb - nonuniform
        F: Fourier coeff of  function f sampled at equally spaced
        ff: values of function evaluate at points (lb_i, th_j)
        U_x has dimension N by 3;
    */
    void dfs_interp(view_2d<Complex> U, view_2d<Complex> V, view_2d<Complex> W,
     view_1d<Real> th, view_1d<Real> lb, view_r3pts<Real> U_X);

}
#endif
