#include <cstdio>
#include <sstream>
#include <fftw3.h>
#include "dfs_velocity.hpp"


using namespace SpherePoisson;

Real test_mattrig(int nrows, int ncols);

int main(int argc, char * argv[])
{
    Kokkos::initialize(argc, argv);
    {
        int nrows = 12;
        int ncols = 24;
        Real err;
       
        err =  test_mattrig(nrows,  ncols);
        
        if(err == 0)
        {
            std::cout<<"Trig multiplication function is correct \n";
        }
        else{
            std::cout<<"Trig multiplication function incorrect \n";
            exit(-1);
        }


        
        
    }
    // kokkos scope
    Kokkos::finalize();
    return 0;
}

Real test_mattrig(int nrows, int ncols)
{
    view_2d<Complex> mat("matrix", nrows, ncols);
    view_2d<Complex> matb("matrix", nrows, ncols);
    view_2d<Complex> mata("matrix", nrows, ncols);
    view_2d<Complex> mata2("matrix", nrows, ncols);
      Kokkos::parallel_for("in",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrows, ncols}), KOKKOS_LAMBDA (const int i, const int j) {
            matb(i,j) = 1.0;
      });

    multi_trig(matb, mat, "cos");
    
    Complex sum_2 = 0.0;
    // KOKKOS_LAMBDA macro includes capture-by-value specifier [=].
    Kokkos::parallel_reduce ("Reduction", ncols, KOKKOS_LAMBDA (const int j, Complex& update) {
        Complex col_sum=0;
        Complex val = ((j==0)||(j==ncols-1)) ?
                0.5 : 1.0;
        for(int i =0; i<nrows; i++){
            col_sum += (mat(i,j) - val); 
        }
        update += col_sum;
   }, sum_2);


      Kokkos::parallel_for("in",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrows, ncols}), KOKKOS_LAMBDA (const int i, const int j) {
            mata2(i,j) = 1.0;
      });

    multi_trig(mata2, mata, "sin");
    
    Complex sum_2b = 0.0;
    // KOKKOS_LAMBDA macro includes capture-by-value specifier [=].
    Kokkos::parallel_reduce ("Reduction", ncols, KOKKOS_LAMBDA (const int j, Complex& update) {
        Complex col_sum=0;
        Complex mval = 0;
    
        if(j==0)
        {
            mval = Complex(0.0, -0.5);
        }
        else if(j==ncols-1)
        {
            mval = Complex(0.0, 0.5);
        }
        for(int i =0; i<nrows; i++){
            col_sum += (mata(i,j) - mval); 
        }
        update += col_sum;
   }, sum_2b);

    return abs(sum_2) + abs(sum_2b);
}