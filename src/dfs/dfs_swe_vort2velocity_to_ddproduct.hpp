#ifndef DFS_SWE_VORT2VELOCITY_TO_DDPRODUCT_HPP
#define DFS_SWE_VORT2VELOCITY_TO_DDPRODUCT_HPP

#include "dfs_config.hpp"
#include <KokkosBlas1_axpby.hpp>
#include "dfs_laplacian_new.hpp"
#include "dfs_rhs_new.hpp"
#include "dfs_solve_new.hpp"
#include "dfs_velocity.hpp"
#include "dfs_doubling.hpp"
#include "dfs_interpolation.hpp"
#include "dfs_surface_grad.hpp"

namespace SpherePoisson{
/*
     Inputs:
     xyz_particles : N by 3 views with components (x,y,z) of on the sphere
        in cartesian coordinates;

     vort: vorticity sampled on a 2d colatitude/longitude grid
        colatitude = (pi*j)/(nrows-1), j=0, 1, ..., nrows-1
        longitude = 2*pi * k/ncols, k=0, 1, ..., ncols-1
            we exclude the last point for becuase of FFT
    Goal:
        Given vorticity on the grid compute velocity U_xyz(u,v,w)
        for each particle alpha(x,y,z).

    Steps:
       (1) Solve Poisson equation in Fourier space to obtain the stream function
       using the vorticity as the source (right handside of the poisson equation)

       (2) Compute the rotation of the stream function in Fourier space
        to obtain Fourier coefficients of U(u, v, w)

       (3) Perform the Non uniform FFT on the Fourier coefficients of U(u,v,w)
       to evaluate U_x(u,v,w) at the xyz_particles.

       (4) The velocity  is then passed to the lpm code
       
       (5) The Fourier coefficients of the velocity on the grid is also passed as an argument 
       to be used for the fast omputation of the double dot product for the swe
       
*/
    void dfs_vort2velocity_swe(view_r3pts<Real> xyz_particles, view_1d<Real> vort, view_r3pts<Real>U_X,
    view_3d<Complex> Ucoeffs);

    /*
        Inputs:
        (1) xyz_particles: An N by 3 view of the particles positions alpha(x,y,z)
        (2) Ucoeffs: An M by M by 3 view of the Fourier coefficients of the velocity
            computed using the DFS solver
        (3) ddproduct_particles: A N by 3 view of the double dot product of the
        velocity computed at the particles alpha(xyz_particles)
        Goal:
        The goal of this function is to compute the double dot product of the velocity
        at the partiles from the Fourier coefficients of the current velocity. 
        These coefficients are computed using the DFS fast solver implemented in the functon 
        "dfs_vort2velocity_swe" above.
    */
   void velocity_coeffs_to_ddproduct(view_r3pts<Real> xyz_particles, 
   view_3d<Complex> Ucoeffs, view_1d<Real> ddproduct_particles);
}
#endif