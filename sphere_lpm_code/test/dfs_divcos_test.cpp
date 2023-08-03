#include <cstdio>
#include <sstream>
#include <fftw3.h>
#include "dfs_surface_grad.hpp"


using namespace SpherePoisson;

Real test_div_cos(int nrows, int ncols);

int main(int argc, char * argv[])
{
    Kokkos::initialize(argc, argv);
    {
        int nrows = 12;
        int ncols = 24;
        Real err;
       
        err =  test_div_cos(nrows,  ncols);
        
        if(err == 0)
        {
            std::cout<<"Division by cosine correct \n";
        }
        else{
            std::cout<<"Division by cosine incorrect \n";
            exit(-1);
        }


        
        
    }
    // kokkos scope
    Kokkos::finalize();
    return 0;
}

Real test_div_cos(int nrows, int ncols)
{
    view_2d<Complex> mat("matrix", nrows, ncols);
    view_2d<Complex> matb("matrix", nrows, ncols);
    view_1d<Real> res("res", nrows);

    // manually input  12 entries
    res(0)= 0; res(1) = 2; res(2)= 2; res(3) = 0;
    res(4)= 0; res(5) = 2; res(6)= 2 ; res(7) = 0;
    res(8) = 0; res(9) = 2; res(10) = 2; res(11) = 0;
    //
    Kokkos::parallel_for("in",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrows, ncols}), KOKKOS_LAMBDA (const int i, const int j) {
            matb(i,j) =  1.0;
      });

    div_cos(matb, mat);
    Complex sum_2 = 0.0;
    // KOKKOS_LAMBDA macro includes capture-by-value specifier [=].
    Kokkos::parallel_reduce("Reduction", nrows, KOKKOS_LAMBDA(const int i, Complex& update) {
        Complex col_sum=0;
        Real val = res(i);

        for(int j =0; j<ncols; j++){
            col_sum += (mat(i,j) - val); 
        }
        update += col_sum;
   }, sum_2);

    
    return abs(sum_2);
}