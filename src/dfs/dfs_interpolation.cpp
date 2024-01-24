#include "dfs_interpolation.hpp"

namespace SpherePoisson
{
    // We will use this function interpolate velocity  U_x=(u,v,w) from 
    // the grid to any points on the sphere
    // U_x has size N by 3, where N is the number of particles
     void dfs_interp(view_2d<Complex> U, view_2d<Complex> V, view_2d<Complex> W,
     view_1d<Real> th, view_1d<Real> lb, view_r3pts<Real> U_X)
     {
        Int N1 = U.extent(0);
        Int N2 = U.extent(1);
        Int nrows = th.extent(0);
       
        // rearrange the intput in the form accepted by the finufft library
        view_1d<Complex> cu("cu", nrows);
        view_1d<Complex> uj("uj", N1*N2);
        view_1d<Complex> cv("cv", nrows);
        view_1d<Complex> vj("vj", N1*N2);
        view_1d<Complex> cw("cw", nrows);
        view_1d<Complex> wj("wj", N1*N2);


        // copy sample coefficients in 1d vectors
         Kokkos::parallel_for("initialize",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {N1, N2}), KOKKOS_LAMBDA (const int i, const int j) {
    
            uj(i + j * N1) = U(i,j);
            vj(i + j * N1) = V(i,j);
            wj(i + j * N1) = W(i,j);
            
        });

        // Evaluate NUFFT via the guru interface

        // step 1: create plan
        int64_t Ns [] = {N1, N2};   // N1, N2 as 64-bit int
        Int type = 2, dim = 2, ntrans=1, M = nrows;
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
        finufft_execute(plan, cu.data(), uj.data());
        finufft_execute(plan, cv.data(), vj.data());
        finufft_execute(plan, cw.data(), wj.data());

        //  step 4: copy the results into a Kokkos::View

        Kokkos::parallel_for(M, KOKKOS_LAMBDA (const int i) {
            U_X(i,0) = cu(i).real();
            U_X(i,1) = cv(i).real();
            U_X(i,2) = cw(i).real();
        });
    
        // step 5: when done, free the memory using the 
        // and set pointers to NULL
        finufft_destroy(plan);
        
     }


}