#include <cstdio>
#include <sstream>
#include <fftw3.h>
#include "dfs_velocity.hpp"


using namespace SpherePoisson;

Real test_mat_cos2theta(int nrows, int ncols);

int main(int argc, char * argv[])
{
    Kokkos::initialize(argc, argv);
    {
        int nrows = 12;
        int ncols = 24;
        Real err;
       
        err =  test_mat_cos2theta(nrows,  ncols);
        
        if(err == 0)
        {
            std::cout<<"Derivative function is correct \n";
        }
        else{
            std::cout<<"Derivative function in correct \n";
            exit(-1);
        }


        
        
    }
    // kokkos scope
    Kokkos::finalize();
    return 0;
}

Real test_mat_cos2theta(int nrows, int ncols)
{
    view_2d<Complex> mat("matrix", nrows, ncols);
    view_2d<Complex> matb("matrix", nrows, ncols);
      Kokkos::parallel_for("in",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrows, ncols}), KOKKOS_LAMBDA (const int i, const int j) {
            matb(i,j) = 1.0;
      });

    multi_cos2theta(matb, mat);
    Complex sum_2 = 0.0;
    // KOKKOS_LAMBDA macro includes capture-by-value specifier [=].
    Kokkos::parallel_reduce ("Reduction", nrows, KOKKOS_LAMBDA (const int i, Complex& update) {
        Complex col_sum=0;
        Complex val = ((i==0)||(i==1)||(i==nrows-2)||(i==nrows-1)) ?
                0.5 : 1.0;
        for(int j =0; j<ncols; j++){
            col_sum += (mat(i,j) - val); 
        }
        update += col_sum;
   }, sum_2);

    return abs(sum_2);
}