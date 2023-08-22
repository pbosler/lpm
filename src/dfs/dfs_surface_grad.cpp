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

    // Computing surface gradient in values
    void dfs_grad_u(view_2d<Complex> du_dtheta, view_2d<Complex> du_dlambda, view_2d<Real> du_dx,
    view_2d<Real> du_dy, view_2d<Real> du_dz)
    {
        Int nrows = du_dtheta.extent(0);
        Int ncols = du_dlambda.extent(1);
        Real alpha = -1;
        Real beta = 1;

        const GridType grid_type = static_cast<GridType>(1);
        view_2d<Complex> lhs("lhs", nrows, ncols);
        view_2d<Complex> lhs_a("lhs", nrows, ncols);
        view_2d<Complex> rhs("rhs", nrows, ncols);
        view_2d<Complex> tmp("rhs", nrows, ncols);
        view_1d<Complex> cn("cn", nrows);
        interp_shifts(grid_type, cn);

        // compute du/dx (du_dx)
        multi_trig(du_dlambda, lhs_a, "sin");
        divide_sin(lhs_a, lhs);

        multi_trig(du_dtheta, rhs, "cos");
        multi_cos(rhs, tmp);
        KokkosBlas::axpby(alpha, lhs, beta, tmp);  
        coeffs2valsDbl(grid_type, cn, tmp, du_dx);


        // compute du/dy (du_dy)
        multi_trig(du_dlambda, lhs_a, "cos");
        div_cos(lhs_a, lhs);

        multi_trig(du_dtheta, rhs, "sin");
        multi_cos(rhs, tmp);
        alpha = 1.0;
        KokkosBlas::axpby(alpha, lhs, beta, tmp);  
        coeffs2valsDbl(grid_type, cn, tmp, du_dy);

        // compute du/dz (du_dz)
        multi_sin(du_dtheta, tmp);
        KokkosBlas::scal(tmp, -1, tmp);
        coeffs2valsDbl(grid_type, cn, tmp, du_dz);

    }
    
    // double dot product of the surface gradient on the sphere
     void double_dot_gradU(view_2d<Complex> U, view_2d<Complex> V, view_2d<Complex> W, view_1d<Real> gradU_ddot)
     {
        Int nrows = U.extent(0);
        Int ncols = U.extent(1);
        
         // differentiation operators
        view_1d<Complex> d_theta("d/dtheta", nrows);
        view_1d<Complex> d_lambda("d/dlambda", ncols);
        dfs_diff_operators(d_theta, d_lambda);
     
        // differentiate component u
        view_2d<Complex> du_theta("du/dtheta", nrows, ncols);
        view_2d<Complex> du_lambda("du/dlambda", nrows, ncols);
        dfs_diff_lat_lon(U, d_theta, d_lambda, du_theta, du_lambda);
        // surface gradient for component u
        view_2d<Real> du_dx("du/dx", nrows, ncols);
        view_2d<Real> du_dy("du/dy", nrows, ncols);
        view_2d<Real> du_dz("du/dz", nrows, ncols);
        dfs_grad_u(du_theta, du_lambda, du_dx, du_dy, du_dz);

        // differentiate component v
        view_2d<Complex> dv_theta("dv/dtheta", nrows, ncols);
        view_2d<Complex> dv_lambda("dv/dlambda", nrows, ncols);
        dfs_diff_lat_lon(V, d_theta, d_lambda, dv_theta, dv_lambda);
        // surface gradient for component v
        view_2d<Real> dv_dx("dv/dx", nrows, ncols);
        view_2d<Real> dv_dy("dv/dy", nrows, ncols);
        view_2d<Real> dv_dz("dv/dz", nrows, ncols);
        dfs_grad_u(dv_theta, dv_lambda, dv_dx, dv_dy, dv_dz);
  
        // differentiate component w
        view_2d<Complex> dw_theta("dw/dtheta", nrows, ncols);
        view_2d<Complex> dw_lambda("dw/dlambda", nrows, ncols);
        dfs_diff_lat_lon(W, d_theta, d_lambda, dw_theta, dw_lambda);
        // surface gradient for component v
        view_2d<Real> dw_dx("dv/dx", nrows, ncols);
        view_2d<Real> dw_dy("dv/dy", nrows, ncols);
        view_2d<Real> dw_dz("dv/dz", nrows, ncols);
        dfs_grad_u(dw_theta, dw_lambda, dw_dx, dw_dy, dw_dz);


    // computing the double dot of the velocity tensor
    
   Kokkos::parallel_for("doubledot",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrows, ncols}), KOKKOS_LAMBDA (const int i, const int j) {
        
       
        Real tmp1 = du_dx(i,j)*du_dx(i,j) +  dv_dy(i,j)*dv_dy(i,j) +  dw_dz(i,j)*dw_dz(i,j);
        Real tmp2 = 2.0 * (du_dy(i,j)*dv_dx(i,j) + du_dz(i,j)*dw_dx(i,j)
                        + dv_dz(i,j)*dw_dy(i,j));
        gradU_ddot(i*ncols+j) = tmp1 + tmp2;

     });
}  

    // Computing the double dot 
    void compute_ddot_X(view_2d<Complex> U, view_2d<Complex> V, view_2d<Complex> W, view_r3pts<Real> X,view_1d<Real> uddot_x)
    {
        Int dnrows = U.extent(0);
        Int ncols = U.extent(1);
        Int nrows = dnrows/2 + 1;

        view_1d<Real> uddot("uddot", nrows+ncols);     // double dot on the grid
        view_2d<Complex> Uddot("uddot coeffs", dnrows, ncols);
        view_1d<Complex> cn("cn", dnrows);

        // Compute the double dot product on the grid
         double_dot_gradU(U, V,  W, uddot);
    
        // compute the Fourier coefficients of the double dot product
        GridType gridtype = static_cast<GridType>(1);
        interp_shifts(gridtype,  cn);
        vals2CoeffsDbl(gridtype, cn, uddot, Uddot);

        // Prepare for interpolation
        Int M = X.extent(0);
        view_1d<Complex> coeffs("uddot coeffs", dnrows*ncols);

        // particle positions in spherical coordinates
        view_1d<Real> th("theta", M);
        view_1d<Real> lb("lambda", M);
        view_1d<Complex> cu("result", M);

        Kokkos::parallel_for(M, KOKKOS_LAMBDA(Int i){
            th(i) = atan2(sqrt(X(i,0)*X(i,0) + X(i,1)*X(i,1)), X(i,2));
            lb(i) = atan2(X(i,1), X(i, 0));

        });


        // copy sample coefficients in 1d vectors
         Kokkos::parallel_for("initialize",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {dnrows, ncols}), KOKKOS_LAMBDA (const int i, const int j) {
    
            coeffs(i + j * dnrows) = Uddot(i,j);
        });

        // Interpolating to the particles
        // step 1: create plan
        int64_t Ns [] = {dnrows, ncols};   
        Int type = 2, dim = 2, ntrans=1;
        finufft_plan plan;
        Int ier = finufft_makeplan(type, dim, Ns, +1, ntrans, 
                    1e-15, &plan, NULL);

        
        // step 2: send in pointers to M nonuniform points
        // just x, y, in this case) ....
        // we make sure x and y are not change after this step before 
        // completing the process
        finufft_setpts(plan, M, th.data(), lb.data(), NULL, 0, NULL, NULL, NULL);

        // step 3: do the planned transform to the cstrength
        // data, output to F...
        finufft_execute(plan, cu.data(), coeffs.data());

        //  step 4: copy the results into a Kokkos::View

        Kokkos::parallel_for(M, KOKKOS_LAMBDA (const int i) {
            uddot_x(i) = cu(i).real();
            
        });
    
        // step 5: when done, free the memory using the 
        // and set pointers to NULL
        finufft_destroy(plan);
    }

} 
