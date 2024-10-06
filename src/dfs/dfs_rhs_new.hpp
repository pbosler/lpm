#ifndef DFS_RHS_NEW_HPP
#define DFS_RHS_NEW_HPP
#include <fftw3.h>
#include "dfs_config.hpp"
#include <KokkosBlas3_gemm.hpp>
#include "dfs_doubling.hpp"

namespace SpherePoisson {
    // function to compute       sshift necessary to interpolate
    // in the interval [-pi, pi] by [0, 2pi]
    void interp_shifts(const GridType grid_type, view_1d<Complex> cn);

    // Computes the fftshift for the case where both
    // nrows and ncols is even
     void fftshift(view_2d<Complex> data);

    // Function for computing the bivariate Fourier series
    // of function defined on the sphere.
    // Steps:
    // (i) Apply the grid reflection to double up the function
    //  (ii) Use FFTW3 to do FFT on the double up grid
    // (iii) Perform bivariate fftshift
    // Shift the coefficients for interpolation in
    // the interval [-pi, pi] \times [0, 2pi]

    void vals2CoeffsDbl(GridType grid_type, view_1d<Complex> cn, view_1d<Real> f, view_2d<Complex>F);

    //  Function for scalaring the right handside of a poisson equation
    // on the sphere by sin^2 theta
    // This computation represnts multiplying bothsides of the equation by
    // sin^2 theta
    void scalerhs(view_2d<Complex> rhs, view_2d<Complex> nrhs);

    void indices_split(view_1d<int>io, view_1d<int>ie);

    // Function for splitting the right handside of the
    // poisson equation into even and odd components
    void splitrhs(view_1d<int>ie, view_1d<int>io, view_2d<Complex> rhs, view_2d<Complex> rhse, view_2d<Complex> rhso);




    // Compute the Fourier coefficients of right handside of poisson
    // Problem on the sphere, scales it by sin^2, and splits it
    // on even and parts to reduce computational complexity
    // by a factor of two
    void poisson_rhs(GridType grid_type, view_1d<Int>ie, view_1d<Int>io, view_1d<Complex>cn, view_1d<Real> f, view_2d<Complex> rhso, view_2d<Complex> rhse);


    // Compute values of a function on the sphere give
    // Fourier coefficients
    void coeffs2valsDbl(GridType grid_type, view_1d<Complex> cn, view_2d<Complex> F, view_2d<Real> f);


}
#endif
