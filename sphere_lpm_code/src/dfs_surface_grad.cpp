#include "dfs_surface_grad.hpp"


namespace SpherePoisson
{
    // sin(theta) * A
    void multi_sin(view_2d<Complex> mat, view_2d<Complex> res)
    {
        Int nrows = mat.extent(0);
        Int ncols = mat.extent(1);
        Complex vala = Complex(0,0.5);
        Complex valb = -Complex(0,0.5);

        Kokkos::parallel_for("in",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrows, ncols}), KOKKOS_LAMBDA (const int i, const int j) {
            if(i == 0)
            {
                res(i,j) = vala * mat(i+1,j);
            }
            else if(i == nrows-1)
            {
                res(i,j) = valb * mat(i-1,j);
            }
            else
            {
                res(i,j) = valb*mat(i-1, j) +  vala*mat(i+1,j);
            }
        });
    }


    // differential operators
    void dfs_diff_operators(view_1d<Complex> d_theta, view_1d<Complex> d_lambda)
     {
        Int n = d_theta.extent(0);
        Int m = d_lambda.extent(0);
        Real wx = -n/2.0;
        Real wy = -m/2.0;

         // derivative with respect to theta
        Kokkos::parallel_for(n-1, [=](Int i){
            d_theta(i+1) = Complex(0, wx + i + 1);
        });

        // derivative with respect to lambda
        Kokkos::parallel_for(m-1, [=](Int i){
            d_lambda(i+1) = Complex(0, wy + i +1 );
        });
     }

     // division by cosine 
     void div_cos(view_2d<Complex> mat, view_2d<Complex> res)
     {
        Int nrows = mat.extent(0);
        Int ncols = mat.extent(1);
        Real val = 0.5;

           // systematically compute mat/sin(theta)
            Kokkos::parallel_for(ncols, KOKKOS_LAMBDA(Int k){
               
               // forward propagation we compute solution
               // for odd i
                res(1, k) = mat(0, k)/val;
                for(int i = 2; i < nrows-1; i += 2)
                {
                    res(i+1,k) = (mat(i, k) - val * res(i-1, k))/val;
                }
                

                // backward propagation we compute solution 
                // for even i
                res(nrows-2, k) = mat(nrows-1, k) / val;
                for(int i=nrows-3; i > 0; i -= 2)
                {
                    res(i-1, k) = (mat(i, k) - val*res(i+1,k)) / val;
                }
                
            });
     }

    
    // derivative
   void dfs_diff_lat_lon(view_2d<Complex> u,view_1d<Complex> d_theta,view_1d<Complex> d_lambda, view_2d<Complex> du_theta, view_2d<Complex> du_lambda)
    {
        Int nrows = u.extent(0);
        Int ncols = u.extent(1);

         Kokkos::parallel_for("differentiate",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrows, ncols}), KOKKOS_LAMBDA (const int i, const int j) {
            du_theta(i, j) = u(i,j) * d_theta(i);
            du_lambda(i, j) = u(i,j) * d_lambda(j);
        });
    }

    // surface gradient of a component
    void dfs_grad_u(view_2d<Complex> du_dtheta, view_2d<Complex> du_dlambda, view_2d<Complex> du_dx,
    view_2d<Complex> du_dy, view_2d<Complex> du_dz)
    {
        Int nrows = du_dtheta.extent(0);
        Int ncols = du_dlambda.extent(1);
        Real alpha = -1;
        Real beta = 1;

        view_2d<Complex> lhs("lhs", nrows, ncols);
        view_2d<Complex> lhs_a("lhs", nrows, ncols);
        view_2d<Complex> rhs("rhs", nrows, ncols);

        // compute du/dx (du_dx)
        multi_trig(du_dlambda, lhs_a, "sin");
        divide_sin(lhs_a, lhs);

        multi_trig(du_dtheta, rhs, "cos");
        multi_cos(rhs, du_dx);
        KokkosBlas::axpby(alpha, lhs, beta, du_dx);  

        // compute du/dy (du_dy)
        multi_trig(du_dlambda, lhs_a, "cos");
        div_cos(lhs_a, lhs);

        multi_trig(du_dtheta, rhs, "sin");
        multi_cos(rhs, du_dy);
        alpha = 1.0;
        KokkosBlas::axpby(alpha, lhs, beta, du_dy);  

        // compute du/dz (du_dz)
        multi_sin(du_dtheta, du_dz);
        KokkosBlas::scal(du_dz, -1, du_dz);

    }
    
    // double dot product of the surface gradient on the sphere
     void double_dot_gradU(view_2d<Complex> u, view_2d<Complex> v, view_2d<Complex> w, view_2d<Complex> gradU_ddot)
     {
        Int nrows = u.extent(0);
        Int ncols = u.extent(1);
        
        // differentiation operators
        view_1d<Complex> d_theta("d/dtheta", nrows);
        view_1d<Complex> d_lambda("d/dlambda", ncols);
        dfs_diff_operators(d_theta, d_lambda);
     
        // differentiate component u
        view_2d<Complex> du_theta("du/dtheta", nrows, ncols);
        view_2d<Complex> du_lambda("du/dlambda", nrows, ncols);
        dfs_diff_lat_lon(u, d_theta, d_lambda, du_theta, du_lambda);
  
        // differentiate component v
        view_2d<Complex> dv_theta("dv/dtheta", nrows, ncols);
        view_2d<Complex> dv_lambda("dv/dlambda", nrows, ncols);
        dfs_diff_lat_lon(v, d_theta, d_lambda, dv_theta, dv_lambda);
  
        // differentiate component w
        view_2d<Complex> dw_theta("dw/dtheta", nrows, ncols);
        view_2d<Complex> dw_lambda("dw/dlambda", nrows, ncols);
        dfs_diff_lat_lon(w, d_theta, d_lambda, dw_theta, dw_lambda);

        // surface gradient for component u
        view_2d<Complex> Du_dx("du/dx", nrows, ncols);
        view_2d<Complex> Du_dy("du/dy", nrows, ncols);
        view_2d<Complex> Du_dz("du/dz", nrows, ncols);
        dfs_grad_u(du_theta, du_lambda, Du_dx, Du_dy, Du_dz);

        // surface gradient for component v
        view_2d<Complex> Dv_dx("dv/dx", nrows, ncols);
        view_2d<Complex> Dv_dy("dv/dy", nrows, ncols);
        view_2d<Complex> Dv_dz("dv/dz", nrows, ncols);
        dfs_grad_u(dv_theta, dv_lambda, Dv_dx, Dv_dy, Dv_dz);

        // surface gradient for component v
        view_2d<Complex> Dw_dx("dv/dx", nrows, ncols);
        view_2d<Complex> Dw_dy("dv/dy", nrows, ncols);
        view_2d<Complex> Dw_dz("dv/dz", nrows, ncols);
        dfs_grad_u(dw_theta, dw_lambda, Dw_dx, Dw_dy, Dw_dz);


    // computing the double dot of the velocity tensor
    Kokkos::parallel_for("doubledot",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrows, ncols}), KOKKOS_LAMBDA (const int i, const int j) {

        Complex tmp1 = Du_dx(i,j)*Du_dx(i,j) +  Dv_dy(i,j)*Dv_dy(i,j) +  Dw_dz(i,j)*Dw_dz(i,j);
        Complex tmp2 = 2.0 * (Du_dy(i,j)*Dv_dx(i,j) + Du_dz(i,j)*Dw_dx(i,j)
                        + Dv_dz(i,j)*Dw_dy(i,j));
        gradU_ddot(i,j) = tmp1 + tmp2;
          
     });
}  
} 