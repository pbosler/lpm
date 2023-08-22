#include "dfs_vort2velocity.hpp"

namespace SpherePoisson {
     void dfs_vort_2_velocity(view_r3pts<Real> X, view_1d<Real> vort,view_r3pts<Real> U_X)
     {

        // Step 1: solve for the streafunction in Fourier space
       
        /* Laplacian operator in Fourier space
         It is split up into even and odd modes to 
          accelerate computation.
          Best Idea is to compute this one and reuse, we keep here for now.
          */
        Int ncols = abs(sqrt(2*vort.extent(0)+1))-1;
        Int nrows = ncols/2 + 1;
        Int dnrows = ncols;   
        Real Kappa = 0;    // For poisson Kappa=0 else Helmholtz
        view_2d<Real> Lo_matrix("odd modes", nrows-1, nrows-1);  
        view_2d<Real> Le_matrix("even modes", nrows-1, nrows-1);

        even_odd_laplacian_matrix(Lo_matrix, Le_matrix, Kappa);

        /* compute the Fourier coefficients
         of the right handside and separate even mode from
         odd modes.
         */
        
        GridType grid_type = static_cast<GridType>(1);
       
        view_2d<Complex> vort_rhso("Coefficients",nrows-1, ncols); // odd modes
        view_2d<Complex> vort_rhse("Coefficients",nrows-1, ncols); // even modes
        view_1d<Int>io("even indices", nrows-1);
        view_1d<Int>ie("odd indices", nrows-1);
        view_1d<Complex>cn("shifts", dnrows);


        indices_split(io, ie);
        interp_shifts(grid_type, cn);
        poisson_rhs(grid_type, ie, io, cn, vort, vort_rhso, vort_rhse);

        /*
            Sove the Fourier coefficients of the stream function
        */
        view_2d<Complex> stream_fun("sfun", dnrows, ncols);
        solver(Le_matrix, Lo_matrix, vort_rhse,  vort_rhso, stream_fun, io, ie, Kappa);

        // Step 2: compute the rotation of the stream function in Fourier space
        /*
            The result are components of the velocity U_X in Fourier space
        */
        view_2d<Complex> U("component_u", dnrows, ncols);
        view_2d<Complex> V("component_v", dnrows, ncols);
        view_2d<Complex> W("component_w", dnrows, ncols);
        
        velocity_on_grid(stream_fun,  U, V, W);

        // Evaluate the velocity on the particles
        /*
            transform the particles cartesian coordinates to 
            spherical coordinates
        */
        Int N = X.extent(0);        // N = Number of particles
        view_1d<Real> co_lat("colat", N);    
        view_1d<Real> lon("lon", N);

        Kokkos::parallel_for(N, KOKKOS_LAMBDA(Int i){
            co_lat(i) = atan2(sqrt(X(i,0)*X(i,0) + X(i,1)*X(i,1)), X(i,2));
            lon(i) = atan2(X(i,1), X(i, 0));

        });

        /*
            Evaluate the velocity at X.
        */
       dfs_interp(U, V, W, co_lat, lon, U_X);


      




        
     }

    

}
