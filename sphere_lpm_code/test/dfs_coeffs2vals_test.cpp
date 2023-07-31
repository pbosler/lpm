#include "dfs_doubling.hpp"
#include "dfs_laplacian_new.hpp"
#include <cstdio>
#include <sstream>
#include <fftw3.h>
#include "dfs_rhs_new.hpp"
#include "dfs_solve_new.hpp"

using namespace SpherePoisson;

double test_coeffs2vals(int nrows, int ncols);

/* 
This program test whether the functions
for Computing Fourier coefficients.
*/

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        Int nrows = 13;
        Int ncols = 24;

        double err = test_coeffs2vals(nrows, ncols);
        
        if(abs(err) > 6e-15)
        {
            std::cout<<"Error with computing values from Fourier coefficients"<<std::endl;
            std::cout<<"The error is "<<err<<std::endl;
            exit(-1);
        }
        else
        {
            std::cout<<"Values from Fourier coiefficients computed correctly"<<std::endl;
            std::cout<<"The error is "<< err<<std::endl;
        }

    }
    // kokkos scope
    Kokkos::finalize();
    return 0;

}

double test_coeffs2vals(int nrows, int ncols)
{
    Real err=0;
    // Compute the coefficients from sample function
    view_2d<Real> f("rhs", nrows, ncols);
    view_2d<Real> ffinal("rhs", nrows, ncols);
    view_2d<Complex> F("coeffs", 2*(nrows-1), ncols);
    GridType grid_type=static_cast<GridType>(1);
    view_1d<Complex> cn("shifts", 2*(nrows-1));
    interp_shifts(grid_type, cn);

    Kokkos::parallel_for("initialize_u",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrows, ncols}), KOKKOS_LAMBDA (const int i, const int j) {
        f(i,j) = i+j;
        });

    vals2CoeffsDbl(grid_type, cn, f, F);

    coeffs2valsDbl(grid_type, cn, F, ffinal);

    for(int i=0; i<nrows; i++)
    {
        for(int j=0; j<ncols; j++)
        {
            Real tmp = fabs(ffinal(i,j) - f(i,j));            
            err = fmax(tmp, err);
            
        }
    }



    return err;
}