#include <iostream>
#include <sstream>
#include <fftw3.h>
#include "dfs_velocity.hpp"
#include "dfs_test_support.hpp"
#include "dfs_rhs_new.hpp"


using namespace SpherePoisson;

Real test_div_sin(int nrows, int ncols);

int main(int argc, char * argv[])
{
    Kokkos::initialize(argc, argv);
    {
        int nrows = 17;
        int ncols = 32;
        Real err;
       
        err =  test_div_sin(nrows,  ncols);
        
        if(err < 5e-14)
        {
            std::cout<<"velocities are correct\n";
        }
        else{
            std::cout<<"Atleast one of the velocities are incorrect \n";
            exit(-1);
        }


        
        
    }
    // kokkos scope
    Kokkos::finalize();
    return 0;
}

Real test_div_sin(int nrows, int ncols)
{
    Real mean_error;
    Int dnrows = 2*(nrows - 1);
    GridType grid_type = static_cast<GridType>(1);
    view_1d<Real> lat("colatitude", nrows);
    view_1d<Real> lon("longitude", ncols);
    view_1d<Complex> cn("cn", dnrows);
    view_2d<Real> vort("vorticity", nrows, ncols);
    view_2d<Complex> vort_coeff("vorticity_coefficients", dnrows, ncols);
    view_2d<Complex> U("u_coeffs", dnrows, ncols);
    view_2d<Complex> V("v_coeffs", dnrows, ncols);
    view_2d<Complex> W("w_coeffs", dnrows, ncols);
    view_2d<Real> u("u", nrows, ncols);
    view_2d<Real> u_true("u", nrows, ncols);
    view_2d<Real> v("v", nrows, ncols);
    view_2d<Real> v_true("v", nrows, ncols);
    view_2d<Real> w("w", nrows, ncols);
    view_2d<Real> w_true("w", nrows, ncols);

    interp_shifts(grid_type, cn);
    coords(lat, lon);
    my_vort(lat, lon, vort);
    vals2CoeffsDbl(grid_type, cn, vort, vort_coeff);

    velocity_on_grid(vort_coeff,  U, V, W);
    true_velocity(lat, lon, u_true, v_true, w_true);
   
    // compute error for u
    coeffs2valsDbl(grid_type, cn, U, u);
    Real err_u = max_error(u_true, u);

    // compute error for v
    coeffs2valsDbl(grid_type, cn, V, v);
    Real err_v = max_error(v_true, v);
    
    // compute error for v
    coeffs2valsDbl(grid_type, cn, W, w);
    Real err_w = max_error(w_true, w);

    // mean error
    mean_error = (err_u + err_v + err_w)/3;
    std::cout<<"err_u = "<<err_u<<std::endl;
    std::cout<<"err_v = "<<err_v<<std::endl;
    std::cout<<"err_w = "<<err_w<<std::endl;
    return mean_error;
}