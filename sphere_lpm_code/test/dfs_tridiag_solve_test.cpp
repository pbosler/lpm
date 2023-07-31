#include "dfs_doubling.hpp"
#include "dfs_laplacian_new.hpp"
#include <cstdio>
#include <sstream>
#include <fftw3.h>
#include "dfs_rhs_new.hpp"
#include "dfs_solve_new.hpp"

using namespace SpherePoisson;

// Tests the Tridiag solver
int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        int nrows = 12;
        
        // we create an example periodic matrix
        view_1d<Complex> d("rhs", nrows);
        view_2d<Real> A("lhs", nrows, nrows);
        view_1d<Complex> dd("rhs", nrows);
        view_2d<Real> AA("lhs", nrows, nrows);
         view_2d<Real> B("lhs", nrows, nrows);

        Kokkos::parallel_for(nrows, [=](Int i){
            d(i) = 1;
            dd(i) = 1.;
            if(i==0){
                A(i,i) =4.0;        AA(i,i) = 4;
                A(i,1)=-1.0;        AA(i,1) = -1.0;
                A(i,nrows-1)=1.0;
                B(i,i) = 4;
                B(i,1) =-1;
            }  
            else if(i==nrows-1){
                A(i,i) =4.0;        AA(i,i) = 4.0;
                A(i,i-1)=1.0;       AA(i,i-1) = 1.0;
                A(i,0)=-1.0;
                
                B(i,i) = 4.0;
                B(i,i-1) = 1.0;
               
           }
           else{
                A(i,i-1) = 1.0;
                A(i,i) =4.0;
                A(i,i+1)=-1.0;

                B(i,i-1) = 1.0;
                B(i,i) =4.0;
                B(i,i+1)=-1.0;

                AA(i,i-1) = 1.0;
                AA(i,i) =4.0;
                AA(i,i+1)=-1.0;

            }
        });
        
        tridiag_solver(A, d);

        // If it is the correct d(i)=0.24 for i=0, ..., nrows-1
        for(int i=0; i<nrows; i++)
        {
            if(abs(d(i)-0.25)>3e-16)
            {
                std::cout<<"The periodic triadiag_solver not working correctly"<<std::endl;
                exit(-1);
            }
            
        }

        // we also test if A is not periodic
       
        tridiag_solver(AA, dd);

        for(int i=0; i<nrows; i++)
        {
            Complex tmp = 0.0;
            for(int j=0; j<nrows; j++)
            { 
                tmp += B(i,j) * dd(j);
            }
            if(abs(tmp - 1.0) > 3e-16)
            {
                std::cout<<"d at j = "<<i<<"  "<<tmp<<std::endl;
                std::cout<<"The triadiag_solver not working correctly"<<std::endl;
                exit(-1);
            }
        }

    }
    // kokkos scope
    Kokkos::finalize();
    return 0;

}