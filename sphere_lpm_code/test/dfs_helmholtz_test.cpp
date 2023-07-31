#include "dfs_doubling.hpp"
#include "dfs_laplacian_new.hpp"
#include <cstdio>
#include <sstream>
#include <fftw3.h>
#include "dfs_rhs_new.hpp"
#include "dfs_solve_new.hpp"

using namespace SpherePoisson;

double test_Hmatrix(view_2d<Real> He, view_2d<Real> Ho);

/* 
This program test whether the functions
for creating the Helmholtz operators works as intended.
*/

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        int nrows = 12;
        
        // we create an example periodic matrix
        view_2d<Real> He("Le", nrows, nrows);
        view_2d<Real> Ho("Lo", nrows, nrows);
        Real Kappa = 4.0;

        even_odd_laplacian_matrix(Ho, He,Kappa);


        double err = test_Hmatrix(He, Ho);

        if(abs(err) > 6e-15)
        {
            std::cout<<"Error with computing the Helmholtz matrices"<<std::endl;
            exit(-1);
        }
        else
        {
            std::cout<<"Helmholtz matrices computed correctly"<<std::endl;
        }

    }
    // kokkos scope
    Kokkos::finalize();
    return 0;

}

double test_Hmatrix(view_2d<Real> He, view_2d<Real> Ho)
{
    Real err;
    // True even L_matrix
    Real He_temp[12][12] = {
                        {-64, 23.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23.5},
                        {16, -42, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 18.5, -24, 6.5, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 10, -10, 1, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 3.5, 0, -2.5,  0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, -1.0, 6.0, -4.0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, -3.5, 8.0, -3.5, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, -4.0, 6.0, -1.0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, -2.5, 0, 3.5, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0,  0, 1.0, -10.0, 10.0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 6.5, -24.0, 18.5},
                        {16.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14.0, -42.0}
                    };

    // True odd L matrix
    Real Ho_temp[12][12] = {
                {-52.5, 18.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {23.5, -32.5, 10.0, 0, 0, 0, 0, 0, 0, 0, 0,0},
                {0, 14.0, -16.5, 3.5, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 6.5, -4.50, -1.0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 1.0, 3.5, -3.5, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, -2.5, 7.5, -4.0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, -4.0, 7.5, -2.5, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, -3.5, 3.5, 1.0, 0,  0, 0},
                {0, 0, 0, 0, 0, 0, 0, -1.0, -4.5, 6.5, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 3.5, -16.5, 14.0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 10.0, -32.5, 23.5},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18.5, -52.5}
            };

        Real error_He =-99.1;
        for(int i=0; i<12; i++)
        {
            for(int j=0; j<12; j++)
            {
                error_He = fmax(fabs(He_temp[i][j] - He(i,j)), error_He);
               
            }
        }
        
        Real error_Ho = - 99.3;
        for(int i=0; i<12; i++)
        {
            for(int j=0; j<12; j++)
            {
                error_Ho = fmax(fabs(Ho_temp[i][j] - Ho(i,j)), error_Ho);
            }
            
        }
       
        err = error_Ho + error_He;

        return err;
}