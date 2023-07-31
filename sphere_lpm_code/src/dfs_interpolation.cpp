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
        std::vector<std::complex<Real>> cu(nrows);
        std::vector<std::complex<Real>> uj(N1*N2);
        std::vector<std::complex<Real>> cv(nrows);
        std::vector<std::complex<Real>> vj(N1*N2);
        std::vector<std::complex<Real>> cw(nrows);
        std::vector<std::complex<Real>> wj(N1*N2);
        std::vector<Real> x(nrows);
        std::vector<Real> y(nrows);

        // pointers to the locations, this necessary to copy fast
        std::vector<std::complex<Real>>* cu_ptr = &(cu);
        std::vector<std::complex<Real>>* uj_ptr = &(uj);
        std::vector<std::complex<Real>>* cv_ptr = &(cv);
        std::vector<std::complex<Real>>* vj_ptr = &(vj);
        std::vector<std::complex<Real>>* cw_ptr = &(cw);
        std::vector<std::complex<Real>>* wj_ptr = &(wj);
        std::vector<Real>* x_ptr = &(x);
        std::vector<Real>* y_ptr = &(y);

        // copy evaluation point into 1d vectors
        // interface only accepts 1d vectors
        Kokkos::parallel_for(nrows, [=](Int i){
            (*x_ptr)[i] = th(i);
            (*y_ptr)[i] = lb(i);
        });

        // copy sample coefficients in 1d vectors
         Kokkos::parallel_for("initialize",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {N1, N2}), KOKKOS_LAMBDA (const int i, const int j) {
    
            (*uj_ptr)[i + j* N2] = U(i,j);
            (*vj_ptr)[i + j* N2] = V(i,j);
            (*wj_ptr)[i + j* N2] = W(i,j);
            
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
        finufft_setpts(plan, M, &x[0], &y[0], NULL, 0, NULL, NULL, NULL);

        // step 3: do the planned transform to the cstrength
        // data, output to F...
        finufft_execute(plan, &cu[0], &uj[0]);
        finufft_execute(plan, &cv[0], &vj[0]);
        finufft_execute(plan, &cw[0], &wj[0]);

        //  step 4: copy the results into a Kokkos::View

        Kokkos::parallel_for(M, KOKKOS_LAMBDA (const int i) {
            U_X(i,0) = (*cu_ptr)[i].real();
            U_X(i,1) = (*cv_ptr)[i].real();
            U_X(i,2) = (*cw_ptr)[i].real();
        });
    
        // step 5: when done, free the memory using the 
        // and set pointers to NULL
        finufft_destroy(plan);
        uj_ptr = NULL;
        vj_ptr = NULL;
        wj_ptr = NULL;
        x_ptr = NULL;
        y_ptr = NULL;
        cu_ptr = NULL;  
        cv_ptr = NULL; 
        cw_ptr = NULL; 
     }


}