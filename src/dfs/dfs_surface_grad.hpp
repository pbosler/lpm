#ifndef DFS_SURFACE_GRAD_HPP
#define DFS_SURFACE_GRAD_HPP

#include <finufft.h>
#include "dfs_velocity.hpp"
#include<KokkosBlas3_gemm.hpp>
#include<KokkosBlas1_scal.hpp>
#include "dfs_doubling.hpp"
#include "dfs_rhs_new.hpp"
#include "dfs_config.hpp"


namespace SpherePoisson {
    // sin(theta) * A
     void multi_sin(view_2d<Complex> mat, view_2d<Complex> res);

    // precompute differentiation Matrices for reuse
   void dfs_diff_operators(view_1d<Complex> d_theta, view_1d<Complex> d_lambda);

   /*
    optimal division by cosine in Fourier space
    mat refers to any nrows by ncols matrix of Fourier
    coeffiecients.
    cos(theta) = nrows by nrows matrix
    */
   void div_cos(view_2d<Complex> mat, view_2d<Complex> res);

    // differentiate a vector component by theta and lambda
    // where theta - colatitude, lambda -longitude
    void dfs_diff_lat_lon(view_2d<Complex> u,view_1d<Complex> d_theta,view_1d<Complex> d_lambda, view_2d<Complex> du_theta, view_2d<Complex> du_lambda);

    /*
        Computes the surface gradient of a scalar (vector component)
        on the sphere in cartesian cooordinates
    */
    void dfs_grad_u(view_2d<Complex> du_dtheta, view_2d<Complex> du_dlambda, view_2d<Real> du_dx,
    view_2d<Real> du_dy, view_2d<Real> du_dz);

    /*
        Computes the double dot product of the velocity surface gradient tensor define on a co-latitude
        longitude grid in Fourier space. The 3D velocity U has components u, v, w.
        Here:
        -gradU_ddot = double dot product of the velocity surface gradient tensor
    */
   void double_dot_gradU(view_2d<Complex> U, view_2d<Complex> V, view_2d<Complex> W, view_1d<Real> gradU_ddot);

   /*
        Compute double product of the
   */
  void compute_ddot_X(view_2d<Complex> U, view_2d<Complex> V, view_2d<Complex> W, view_r3pts<Real> X,view_1d<Real> uddot_x);

}
#endif
