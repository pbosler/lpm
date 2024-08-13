#ifndef DFS_NUFTT_HPP
#define DFS_NUFTT_HPP

#include "dfs_config.hpp"
#include <vector>
#include <complex.h>

#ifdef __CUDACC__
#include <cufinufft.h>
#include <cuComplex.h>

#else
#include<finufft.h>
#endif

namespace SpherePoisson{
    class NUFFT{
        private:
        // Struct for FINUFFT with OpenMP
        struct MYFINUFFT_OpenMP {
            static void dfs_execute(view_2d<Complex> U, view_2d<Complex> V, view_2d<Complex> W,
            view_1d<Real> th, view_1d<Real> lb, view_r3pts<Real> U_X)
            {
                Int N1 = U.extent(0);
                Int N2 = U.extent(1);
                Int M = U_X.extent(0);
       
                // rearrange the intput in the form accepted by the finufft library
                view_1d<Complex> cu("cu", M);
                view_1d<Complex> uj("uj", N1*N2);
                view_1d<Complex> cv("cv", M);
                view_1d<Complex> vj("vj", N1*N2);
                view_1d<Complex> cw("cw", M);
                view_1d<Complex> wj("wj", N1*N2);


                // copy sample coefficients in 1d vectors
                Kokkos::parallel_for("initialize",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {N1, N2}), KOKKOS_LAMBDA (Int i, Int j) {
    
                    uj(i + j*N1) = U(i,j);
                    vj(i + j*N1) = V(i,j);
                    wj(i + j*N1) = W(i,j);
            
                });

                // Evaluate NUFFT via the guru interface

                // step 1: create plan
                int64_t Ns [] = {N1, N2};   // N1, N2 as 64-bit int
                Int type = 2, dim = 2, ntrans=1;
        
                finufft_plan plan;
                Int ier = finufft_makeplan(type, dim, Ns, 1, ntrans, 
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
                finufft_destroy(plan);
        
                //  step 4: copy the results into a Kokkos::View
                Kokkos::parallel_for(M, KOKKOS_LAMBDA (const Int i) {
                    U_X(i,0) = cu[i].real();
                    U_X(i,1) = cv[i].real();
                    U_X(i,2) = cw[i].real();
             
                });

            }
        };

        // NUFFT on the GPU
        struct MYCUFINUFFT_GPU{
            static void dfs_execute(view_2d<Complex> U, view_2d<Complex> V, view_2d<Complex> W,
            view_1d<Real> th, view_1d<Real> lb, view_r3pts<Real> U_X)
            {
        #ifdef __CUDACC__
                Int N1 = U.extent(0);
                Int N2 = U.extent(1);
                Int M = U_X.extent(0);
       
                // rearrange the intput in the form accepted by the finufft library
                view_1d<Complex> cu("cu", M);
                view_1d<Complex> uj("uj", N1*N2);
                view_1d<Complex> cv("cv", M);
                view_1d<Complex> vj("vj", N1*N2);
                view_1d<Complex> cw("cw", M);
                view_1d<Complex> wj("wj", N1*N2);


                // copy sample coefficients in 1d vectors
                Kokkos::parallel_for("initialize",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {N1, N2}), KOKKOS_LAMBDA (Int i, Int j) {
    
                    uj(i + j*N1) = U(i,j);
                    vj(i + j*N1) = V(i,j);
                    wj(i + j*N1) = W(i,j);
            
                });

                // Evaluate NUFFT via the guru interface

                // step 1: create plan
                int64_t Ns [] = {N1, N2};   // N1, N2 as 64-bit int
                Int type = 2, dim = 2, ntrans=1;
        
                cufinufft_plan plan;
                cufinufft_makeplan(type, dim, Ns, 1, ntrans, 
                    1e-14, &plan, NULL);

        
                // step 2: send in pointers to M nonuniform points
                // just x, y, in this case) ....
                // we make sure x and y are not change after this step before 
                // completing the process
                cufinufft_setpts(plan, M, th.data(), lb.data(), NULL, 0, NULL, NULL, NULL);

                // step 3: do the planned transform to the cstrength
                // data, output to F...
                cufinufft_execute(plan, cu.data(), uj.data());
                cufinufft_execute(plan, cv.data(), vj.data());
                cufinufft_execute(plan, cw.data(), wj.data());
                cufinufft_destroy(plan);
        
                //  step 4: copy the results into a Kokkos::View
                Kokkos::parallel_for(M, KOKKOS_LAMBDA (const Int i) {
                    U_X(i,0) = cu[i].real();
                    U_X(i,1) = cv[i].real();
                    U_X(i,2) = cw[i].real();
             
                });
        #else
                std::cerr<<"Error:CUFINUFFT requires CUDA support but was not compiled with CUDA."<<std::endl;
        #endif
            }
        };

        public:
            static void dfs_interpolate(view_2d<Complex> U, view_2d<Complex> V, view_2d<Complex> W,
            view_1d<Real> th, view_1d<Real> lb, view_r3pts<Real> U_X)
            {
                #if Kokkos::SpaceAccessibility<Kokkos::Cuda>::accessible
                    MYCUFINUFFT_GPU::dfs_execute(U, V, W, th, lb, U_X);
                
                #else
                    MYFINUFFT_OpenMP::dfs_execute(U, V, W, th, lb, U_X);
                #endif
            }
    };

}
#endif