#ifndef DFS_VORT2VELOCITY_HPP
#define DFS_VORT2VELOCITY_HPP

#include "dfs_config.hpp"
#include <KokkosBlas1_axpby.hpp>
#include "dfs_laplacian_new.hpp"
#include "dfs_rhs_new.hpp"
#include "dfs_solve_new.hpp"
#include "dfs_velocity.hpp"
#include "dfs_doubling.hpp"
#include "dfs_interpolation.hpp"


namespace SpherePoisson {

    /*
     Inputs:
     X: N by 3 views with components (x,y,z) of on the sphere
        in cartesian coordinates;

     vort: vorticity sampled on a 2d colatitude/longitude grid
        colatitude = (pi*j)/(nrows-1), j=0, 1, ..., nrows-1
        longitude = 2*pi * k/ncols, k=0, 1, ..., ncols-1
            we exclude the last point for becuase of FFT
    Goal:
        Given vorticity on the grid compute velocity U_X(u,v,w)
        for each particle X.

    Steps:
       (1) Solve Poisson equation in Fourier space to obtain the stream function
       using the vorticity as the source (right handside of the poisson equation)

       (2) Compute the rotation of the stream function in Fourier space
        to obtain fourier coefficients of U_X(u, v, w)

       (3) Perform the Non uniform FFT on the Fourier coefficients of U_X(u,v,w)
       to evaluate U_x(u,v,w) at the particles  X

       (4) The velocity is then paassed to the lpm code





    */
    void dfs_vort_2_velocity(view_r3pts<Real> X, view_1d<Real> vort,view_r3pts<Real> U_X);


}
#endif
