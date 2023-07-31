#include "dfs_doubling.hpp"
#include "dfs_laplacian_new.hpp"
#include <cstdio>
#include <sstream>
#include <fftw3.h>
#include "dfs_rhs_new.hpp"
#include "dfs_solve_new.hpp"

using namespace SpherePoisson;

double test_helmholtz(int nrows, int ncols);

/* 
This program test the correctness of the Helmholtz solver.
*/

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        Int nrows = 13;
        Int ncols = 24;

        double err = test_helmholtz(nrows, ncols);
        
        if(abs(err) > 6e-15)
        {
            std::cout<<"Error with computing Fourier coefficients"<<std::endl;
            std::cout<<"The error is "<<err<<std::endl;
            exit(-1);
        }
        else
        {
            std::cout<<"Fourier coiefficients computed correctly"<<std::endl;
            std::cout<<"The error is "<< err<<std::endl;
        }

    }
    // kokkos scope
    Kokkos::finalize();
    return 0;

}

double test_helmholtz(int nrows, int ncols)
{
    Real err=0;
    Real max_abs=0.0;
    Int size = nrows * ncols;
    
    Real *utrue = new double[size];
    view_2d<Real> u("solution", nrows, ncols);
    
    FILE* myf; 

    // read the real part of even modes
    myf = fopen("../../datafiles/helmholtz.bin","rb");
    if (nullptr == myf) {
      printf("Could not open file.\n");
      exit(-1);
    }
    else{
        fread(utrue, sizeof(double), size, myf);
        fclose(myf);
    }

    

    // Compute the coefficients from sample function
    view_2d<Real> f("rhs", nrows, ncols);
    view_2d<Complex> F("coeffs", 2*(nrows-1), ncols);
    view_2d<Complex> rhse("rhse", nrows-1, ncols);
    view_2d<Complex> rhso("rhse", nrows-1, ncols);
    GridType grid_type=static_cast<GridType>(1);
    view_1d<Complex> cn("shifts", 2*(nrows-1));
    view_1d<Int> ie("ie", nrows-1);
    view_1d<Int> io("io", nrows-1);
    view_2d<Real> Lo_matrix("Lo", nrows-1, nrows-1);
    view_2d<Real> Le_matrix("Le", nrows-1, nrows-1);
    view_2d<Complex> UU("rhso", 2*(nrows-1), ncols);
    Real Kappa = 4.0;

    interp_shifts(grid_type, cn);
    indices_split(io, ie);
    Kokkos::parallel_for("initialize_u",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrows, ncols}), KOKKOS_LAMBDA (const int i, const int j) {
        f(i,j) = i+j;
        });

    // Prepare lefthand and righthand sides using 
    // spectral discretization
     poisson_rhs(grid_type, ie, io, cn, f, rhso, rhse);
     even_odd_laplacian_matrix(Lo_matrix,  Le_matrix, Kappa);


    // Solve the resulting system using to ontain solution
    solver(Le_matrix, Lo_matrix, rhse, rhso, UU, io, ie, Kappa);

    // Convert back to values
    coeffs2valsDbl(grid_type, cn, UU, u);

    for(int i=0; i<nrows; i++)
    {
        for(int j=0; j<ncols; j++)
        {
            max_abs = fmax(fabs(u(i,j)), max_abs);
            err = fmax(fabs(u(i,j)-utrue[i + j*nrows]), err);
            std::cout<<"Error = "<<err<<std::endl;
            
        }
    
    }
    err = err / max_abs;
    std::cout<<err<<std::endl;



    delete [] utrue;
    return err;
}