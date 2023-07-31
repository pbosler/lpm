#include "dfs_doubling.hpp"
#include "dfs_laplacian_new.hpp"
#include <cstdio>
#include <sstream>
#include <fftw3.h>
#include "dfs_rhs_new.hpp"
#include "dfs_solve_new.hpp"

using namespace SpherePoisson;

double test_rhs(int nrows, int ncols);

/* 
This program test whether the functions
for Computing Fourier coefficients.
*/

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        Int nrows = 13;
        Int ncols = 24;

        double err = test_rhs(nrows, ncols);
        
        if(abs(err) > 6e-5)
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

double test_rhs(int nrows, int ncols)
{
    Real err;
    Int size = (nrows - 1) * ncols;
    
    Real *Xe1 = new double[size];
    Real *Xe2 = new double[size];
     Real *Xo1 = new double[size];
    Real *Xo2 = new double[size];
    FILE* myf; 

    // read the real part of even modes
    myf = fopen("../../datafiles/rhse_real.bin","rb");
    if (nullptr == myf) {
      printf("Could not open file.\n");
      exit(-1);
    }
    else{
        fread(Xe1, sizeof(double), size, myf);
        fclose(myf);
    }

    // read the imaginary part of the even modes
    myf = fopen("../../datafiles/rhse_imag.bin","rb");
    if (nullptr == myf) {
      printf("Could not open file.\n");
      exit(-1);
    }
    else{
        fread(Xe2, sizeof(double), size, myf);
        fclose(myf);
    }


    // read the real part of even modes
    myf = fopen("../../datafiles/rhso_real.bin","rb");
    if (nullptr == myf) {
      printf("Could not open file.\n");
      exit(-1);
    }
    else{
        fread(Xo1, sizeof(double), size, myf);
        fclose(myf);
    }

    // read the imaginary part of the odd modes
    myf = fopen("../../datafiles/rhso_imag.bin","rb");
    if (nullptr == myf) {
      printf("Could not open file.\n");
      exit(-1);
    }
    else{
        fread(Xo2, sizeof(double), size, myf);
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

    interp_shifts(grid_type, cn);
    indices_split(io, ie);
    Kokkos::parallel_for("initialize_u",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrows, ncols}), KOKKOS_LAMBDA (const int i, const int j) {
        f(i,j) = i+j;
        });

     poisson_rhs(grid_type, ie, io, cn, f, rhso, rhse);

    for(int i=0; i<nrows-1; i++)
    {
        for(int j=0; j<ncols; j++)
        {
            Real tmp1 = sqrt(pow(Xe1[i + j*(nrows-1)]- rhse(i,j).real(),2) +
                        pow(Xe2[i + j*(nrows-1)] - rhse(i,j).imag(), 2));
            Real tmp2 = sqrt(pow(Xo1[i + j*(nrows-1)]- rhso(i,j).real(),2) +
                        pow(Xo2[i + j*(nrows-1)] - rhso(i,j).imag(), 2));
            err = fmax(tmp1, tmp2);
            
        }
    }


    delete [] Xe1;
    delete [] Xo1;
    delete [] Xe2;
    delete [] Xo2;
    return err;
}