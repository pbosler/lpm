#include "dfs_doubling.hpp"
#include "dfs_laplacian_new.hpp"
#include <cstdio>
#include <sstream>
#include <fftw3.h>
#include "dfs_rhs_new.hpp"
#include "dfs_solve_new.hpp"

using namespace SpherePoisson;

double test_vals2coeffs(int nrows, int ncols);

/* 
This program test whether the functions
for Computing Fourier coefficients.
*/

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        Int nrows = 13;
        Int ncols = 24;

        double err = test_vals2coeffs(nrows, ncols);
        
        if(abs(err) > 5.5e-15)
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

double test_vals2coeffs(int nrows, int ncols)
{
    Real err=0;
    Int size = 2 * (nrows - 1) * ncols;
    // True even L_matrix
    Real *X1 = new double[size];
    Real *X2 = new double[size];
    FILE* myf; 

    // read the real part
    myf = fopen("../../datafiles/coeffs_real.bin","rb");
    if (nullptr == myf) {
      printf("Could not open file.\n");
      exit(-1);
    }
    else{
        fread(X1, sizeof(double), size, myf);
        fclose(myf);
    }

    // read the imaginary part
    myf = fopen("../../datafiles/coeffs_imag.bin","rb");
    if (nullptr == myf) {
      printf("Could not open file.\n");
      exit(-1);
    }
    else{
        fread(X2, sizeof(double), size, myf);
        fclose(myf);
    }

    // Compute the coefficients from sample function
    view_2d<Real> f("rhs", nrows, ncols);
    view_2d<Complex> F("coeffs", 2*(nrows-1), ncols);
    GridType grid_type=static_cast<GridType>(1);
    view_1d<Complex> cn("shifts", 2*(nrows-1));
    interp_shifts(grid_type, cn);

    Kokkos::parallel_for("initialize_u",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrows, ncols}), KOKKOS_LAMBDA (const int i, const int j) {
        f(i,j) = i+j;
        });

    vals2CoeffsDbl(grid_type, cn, f, F);

    for(int i=0; i<2*(nrows-1); i++)
    {
        for(int j=0; j<ncols; j++)
        {
            Real tmp = sqrt(pow(X1[i + j*ncols]- F(i,j).real(),2) +
                         pow(X2[i + j*ncols] - F(i,j).imag(), 2));
                
            err = fmax(tmp, err);
            
        }
    }


    delete [] X1;
    delete [] X2;
    return err;
}