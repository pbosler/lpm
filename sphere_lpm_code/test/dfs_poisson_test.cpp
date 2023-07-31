#include "dfs_doubling.hpp"
#include "dfs_laplacian_new.hpp"
#include <cstdio>
#include <sstream>
#include <fstream>
#include <fftw3.h>
#include "dfs_rhs_new.hpp"
#include "dfs_solve_new.hpp"

using namespace SpherePoisson;

double test_poisson(int nrows, int ncols);

/* 
This program test the correctness of the poisson solver.
*/

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        Int nrows = 13;
        Int ncols = 24;

        double err = test_poisson(nrows, ncols);
        
        if(abs(err*1e-16) > 6e-15)
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

double test_poisson(int nrows, int ncols)
{
    Real err=0;
    Real max_abs=0.0;
    Int size = nrows * ncols;
    
    
    std::vector<Real> utrue(size);
    std::ifstream ifs("../../datafiles/poisson.bin", std::ios::binary | std::ios::in);
    
    ifs.read(reinterpret_cast<char*>(utrue.data()), (size)*sizeof(double));
    ifs.close();
    

    

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
    view_2d<Real> u("Lo", nrows, ncols);
    view_2d<Real> Le_matrix("Le", nrows-1, nrows-1);
    view_2d<Complex> UU("rhso", 2*(nrows-1), ncols);
    Real Kappa = 0.0;

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

    std::cout<<"finish f\n";
    // Convert back to values
    coeffs2valsDbl(grid_type, cn, UU, u);
    std::cout<<"finish compute\n";
    view_2d<Real>::HostMirror h_u = Kokkos::create_mirror_view( u);
    Kokkos::deep_copy(h_u, u);

    for(int i=0; i<nrows; i++)
    {
        for(int j=0; j<ncols; j++)
        {
            max_abs = fmax(fabs(h_u(i,j)), max_abs);
            err = fmax(fabs(h_u(i,j)-utrue[i + j*nrows]), err);
            std::cout<<"Error = "<<err<<std::endl;
            
        }
    
    }
    err = err / max_abs;


   
    return err;
}