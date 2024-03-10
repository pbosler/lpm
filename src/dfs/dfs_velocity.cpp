#include "dfs_velocity.hpp"

namespace SpherePoisson {
    // Computes first derivative in Fourier space
    void differentiate(view_2d<Complex> d_vort_theta, view_2d<Complex> d_vort_lambda)
    {
        Int nrows = d_vort_theta.extent(0);
        Int ncols = d_vort_theta.extent(1);
        Real wx = -nrows/2.0;
        Real wy = -ncols/2.0;
       
        view_1d<Complex> d_theta("d_theta", nrows);
        view_1d<Complex> d_lambda("derivative", ncols);

        // derivative with respect to theta
        Kokkos::parallel_for(nrows-1, [=](Int i){
            d_theta(i+1) = Complex(0, wx + i + 1);
        });

        // derivative with respect to lambda
        Kokkos::parallel_for(ncols-1, [=](Int i){
            d_lambda(i+1) = Complex(0, wy + i +1 );
        });


        Kokkos::parallel_for("differentiate",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrows, ncols}), KOKKOS_LAMBDA (const int i, const int j) {
            d_vort_theta(i, j) *= d_theta(i);
            d_vort_lambda(i, j) *= d_lambda(j);
        });
    }

    // Routine for multiplying a function by cos(2theta)
    // in Fourier space
     void multi_cos2theta(view_2d<Complex> mat, view_2d<Complex> cos2theta_mat)
     {
       
        Int nrows = mat.extent(0);
        Int ncols = mat.extent(1);
        Kokkos::parallel_for("in",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrows, ncols}), KOKKOS_LAMBDA (const int i, const int j) {
            if(i == 0)
            {
               cos2theta_mat(i,j) = 0.5 * mat(2, j);
            }
            else if(i == 1)
            {
                cos2theta_mat(i,j) = 0.5 * mat(3,j);
            }
            else if(i == nrows-2)
            {
                cos2theta_mat(i,j) = 0.5 * mat(nrows-4, j);
            }
            else if(i == nrows-1)
            {
               cos2theta_mat(i,j) = 0.5 * mat(nrows-3, j);
            }
            else
            {
                cos2theta_mat(i,j) = 0.5 * mat(i-2, j) + 0.5 * mat(i+2,j);
            }
    
        });
     }

     // multiply cos(lambda) / sin(lambda)
     void multi_trig(view_2d<Complex> mat, view_2d<Complex> res, std::string str)
    {
        Int nrows = mat.extent(0);
        Int ncols = mat.extent(1);

         Kokkos::parallel_for("copyU",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrows, ncols}), KOKKOS_LAMBDA (const int i, const int j) {
      
            Complex val = (str=="sin") ? Complex(0,0.5) :Complex(0.5, 0.0);
            Complex valb = (str=="sin") ? -Complex(0,0.5) : Complex(0.5, 0.0);
            if(j == 0)
            {
                res(i,j) = valb * mat(i,j);
            }
            else if(j == ncols-1)
            {
                res(i,j) = val * mat(i,ncols-2);
            }
            else
            {
                res(i,j) = valb * mat(i, j-1) + val * mat(i,j+1);
            }
        });
    }

    // cos(theta) * A
    void multi_cos(view_2d<Complex> mat, view_2d<Complex> res)
    {
        Int nrows = mat.extent(0);
        Int ncols = mat.extent(1);
        Real val = 0.5;

        Kokkos::parallel_for("in",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrows, ncols}), KOKKOS_LAMBDA (const int i, const int j) {
            if(i == 0)
            {
                res(i,j) = val * mat(i+1,j);
            }
            else if(i == nrows-1)
            {
                res(i,j) = val * mat(i-1,j);
            }
            else
            {
                res(i,j) = val * (mat(i-1, j) +  mat(i+1,j));
            }
        });
    }

     void divide_sin(view_2d<Complex> mat, view_2d<Complex> res)
        {
            Int nrows = mat.extent(0);
            Int ncols = mat.extent(1);
            Complex vala = Complex(0.0, 0.5);
            Complex valb = -Complex(0.0, 0.5);

           // systematically compute mat/sin(theta)
            Kokkos::parallel_for(ncols, KOKKOS_LAMBDA(Int k){
               
               // forward propagation we compute solution
               // for odd i
                res(1, k) = mat(0, k)/vala;
                for(int i = 2; i < nrows-1; i += 2)
                {
                    res(i+1,k) = (mat(i, k) + vala * res(i-1, k))/vala;
                }
                

                // backward propagation we compute solution 
                // for even i
                res(nrows-2, k) = mat(nrows-1, k) / valb;
                for(int i=nrows-3; i > 0; i -= 2)
                {
                    res(i-1, k) = (mat(i, k) + valb*res(i+1,k)) / valb;
                }
                
            });

        }

        // Computes velocity component on a grid given vorticity
        void velocity_on_grid(view_2d<Complex> vort, view_2d<Complex> U,  view_2d<Complex> V, view_2d<Complex> W)
        {
            Complex alpha = -1.0;
            Complex beta = -1.0;
            Int nrows = vort.extent(0);
            Int ncols = vort.extent(1);
            view_2d<Complex> tmp("temp", nrows, ncols);
            view_2d<Complex> tmp_th("temp", nrows, ncols);
            view_2d<Complex> tmp_lb("temp", nrows, ncols);
            view_2d<Complex> lhs("LHS", nrows, ncols);
            view_2d<Complex> lhs_a("LHS", nrows, ncols);
            
           KokkosBlas::axpby(Complex(1,0.0), vort, Complex(0.0,0.0), W);  
        
           differentiate(vort, W);
           multi_cos(W, lhs_a);
           divide_sin(lhs_a, lhs);
            
            

            // Compute component u
             multi_trig(lhs, tmp, "cos");
             multi_trig(vort, U, "sin");  //  multi_trig(vort, U, "sin")
    
            // u = -lsh_a - u
            KokkosBlas::axpby(alpha, tmp, beta, U);    
            // Compute v
            multi_trig(lhs, lhs_a, "sin");
            multi_trig(vort, V, "cos");
            beta = 1.0;
            KokkosBlas::axpby(alpha, lhs_a, beta, V);   
  

        }
}