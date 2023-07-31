#include <cstdio>
#include <sstream>
#include <fftw3.h>
#include "dfs_velocity.hpp"


using namespace SpherePoisson;

Complex test_derivative(int nrows, int ncols);

int main(int argc, char * argv[])
{
    Kokkos::initialize(argc, argv);
    {
        int nrows = 12;
        int ncols = 24;
        Complex err;
       
        err =  test_derivative(nrows,  ncols);
        
        if(abs(err) == 0)
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

Complex test_derivative(int nrows, int ncols)
{

    view_2d<Complex> d_vort_theta("dt", nrows, ncols);
    view_2d<Complex> d_vort_lambda("dl", nrows, ncols);
    view_1d<Complex> d_theta("d_theta", nrows);
    view_1d<Complex> d_lambda("derivative", ncols);
    Real wx = -nrows/2.0;
    Real wy = -ncols/2.0;
    // derivative with respect to theta
        Kokkos::parallel_for(nrows-1, [=](Int i){
            d_theta(i+1) = Complex(0, wx + i + 1);
        });

        // derivative with respect to lambda
        Kokkos::parallel_for(ncols-1, [=](Int i){
            d_lambda(i+1) = Complex(0, wy + i + 1);
        });
    Kokkos::parallel_for("differentiate",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrows, ncols}), KOKKOS_LAMBDA (const int i, const int j) {
            d_vort_theta(i, j) = 1;
            d_vort_lambda(i, j) = 1;
    });
    
    differentiate(d_vort_theta, d_vort_lambda);

    Complex sum_1 = 0.0;
    // KOKKOS_LAMBDA macro includes capture-by-value specifier [=].
    Kokkos::parallel_reduce ("Reduction", nrows, KOKKOS_LAMBDA (const int i, Complex& update) {
        Complex row_sum=0;
        for(int j=0; j<ncols; j++){
            row_sum += (d_vort_theta(i,j) - d_theta(i)); 
        }
        update += row_sum;
   }, sum_1);

    Complex sum_2 = 0.0;
    // KOKKOS_LAMBDA macro includes capture-by-value specifier [=].
    Kokkos::parallel_reduce ("Reduction", ncols, KOKKOS_LAMBDA (const int j, Complex& update) {
        Complex col_sum=0;
        for(int i=0; i<nrows; i++){
            col_sum += (d_vort_theta(i,j) - d_lambda(j)); 
        }
        update += col_sum;
   }, sum_2);

    return (sum_2 + sum_1);
}