#include "dfs_doubling.hpp"
#include "dfs_laplacian_new.hpp"
#include <cstdio>
#include <sstream>
#include <fftw3.h>
#include "dfs_rhs_new.hpp"
#include "dfs_solve_new.hpp"

using namespace SpherePoisson;

double test_Lmatrix(view_2d<Real> Le, view_2d<Real> Lo);

/* 
This program test whether the functions
for creating the Laplace operators works as intended.
*/

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        int nrows = 12;
        
        // we create an example periodic matrix
        view_2d<Real> Le("Le", nrows, nrows);
        view_2d<Real> Lo("Lo", nrows, nrows);
        Real Kappa = 0.0;

        even_odd_laplacian_matrix(Lo, Le,Kappa);


        double err = test_Lmatrix(Le, Lo);

        if(abs(err) > 3e-16)
        {
            std::cout<<"Error with computing the Laplacian matrices"<<std::endl;
            exit(-1);
        }
        else
        {
            std::cout<<"Laplacian matrices computed correctly"<<std::endl;
        }

    }
    // kokkos scope
    Kokkos::finalize();
    return 0;

}

double test_Lmatrix(view_2d<Real> Le, view_2d<Real> Lo)
{
    Real err;
    // True even L_matrix
    Real Le_temp[12][12]={{-72,27.50,0,0,0,0,0,0,0,0,0,27.5},
        {18.0, -50.0, 18.0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 22.50, -32.00, 10.50, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 14.0, -18.0, 5.0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 7.50, -8.0, 1.50, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 3.0, -2.0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0.5, 0, 0.5, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, -2.0, 3.0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 1.5, -8.0, 7.50, 0, 0},
         {0,0, 0, 0, 0, 0,  0, 0, 5.0, -18.0, 14.0, 0},
         {0,0, 0, 0, 0, 0, 0, 0, 0, 10.5, -32.0, 22.50},
    {18.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18.0, -50.0}};

    // True odd L matrix
    Real Lo_temp[12][12] = {{-60.5, 22.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {27.5, -40.5, 14.0, 0,  0, 0, 0, 0, 0, 0, 0, 0},
            {0, 18.0, -24.5, 7.5, 0, 0,  0, 0, 0, 0, 0, 0},
            {0, 0, 10.5, -12.5, 3.0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 5.0, -4.50, 0.50, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 1.5, -0.5, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, -0.5, 1.5, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0.50, -4.5, 5.0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0,  0, 3.0, -12.50, 10.50, 0, 0},
            {0, 0, 0, 0, 0, 0,  0, 0,7.5, -24.5, 18.0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 14.0, -40.5, 27.5},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22.5, -60.5},
            };

        Real error_Le =-99.1;
        for(int i=0; i<12; i++)
        {
            for(int j=0; j<12; j++)
            {
                error_Le = fmax(fabs(Le_temp[i][j] - Le(i,j)), error_Le);
               
            }
            
        }
        
        Real error_Lo = - 99.3;
        for(int i=0; i<12; i++)
        {
            for(int j=0; j<12; j++)
            {
                error_Lo = fmax(fabs(Lo_temp[i][j] - Lo(i,j)), error_Lo);
            }
            
        }
       
        err = error_Lo + error_Le;

        return err;
}