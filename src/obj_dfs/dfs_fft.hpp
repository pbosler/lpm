#ifndef DFS_FFT_HPP
#define DFS_FFT_HPP

#include <Kokkos_Core.hpp>
#include <fftw3.h>
#include "dfs_config.hpp"

#ifdef __CUDACC__
#include <cufft.h>
#endif


/* 
 A class defining interfaces for the fast Fourier transform using Kokkos.
 It will select appropriate library depending on Kokkos execution space.
 On cpu it will use FFTW3 and on the GPU it uses CUFFT
*/
namespace SpherePoisson{
template <typename ViewType>
class KokkosFFT{
    public:
        static void dfs_forwardFFT(ViewType data){
            if constexpr(std::is_same<typename ViewType::execution_space,Kokkos::Cuda::value){
                dfs_forwardCUFFT(data);
            } else {
                dfs_forwardFFTW(data);
        }

        }
    private:
        // CPU
        static void forwardFFTW(ViewType data){
            Int Nx = data.extent(0);
            Int Ny = data.extent(1);

            fftw_plan plan;
            fftw_complex *fftw_in, *fftw_out;
            fftw_in  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*(Nx * Ny));
            fftw_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*(Nx * Ny));

            // It should work for every kind of data Layout
            if (data.layout() == Kokkos::LayoutLeft){
                Kokkos::parallel_for(Nx * Ny, KOKKOS_LAMBDA(const Int i){
                    fftw_in[i][0] = data(i  / Ny, i % Ny).real();
                    fftw_in[i][1] = data(i / Ny, i % Ny).imag();
                });
            }
            else{
                Kokkos::parallel_for(Nx * Ny, KOKKOS_LAMBDA(const Int i){
                    fftw_in[i][0] = data(i % Nx, i / Nx).real();
                    fftw_in[i][0] = data(i % Nx, i / Nx).imag();
                });
            }
            Kokkos::fence();    // Making sure it finishes before execute fft
            plan = fftw_plan_dft_2d(Nx, Ny, fftw_in, fftw_out, FFTW_FORWARD, FFTW_ESTIMATE);
            fftw_execute(plan);

            // trasfering the data back to the kokkos views
            if(data.layout() == Kokkos::LayoutLeft){
                Kokkos::parallel_for(Nx * Ny, KOKKOS_LAMBDA(const Int i){
                    data(i / Ny, i % Ny).real(fftw_out[i][0]);
                    data(i / Ny, i % Ny).imag(fftw_out[i][1]);
                });
            }
            else{
                Kokkos::parallel_for(Nx * Ny, KOKKOS_LAMBDA(const Int i){
                    data(i % Nx, i / Nx).real(fftw_out[i][0]);
                    data(i % Nx, i / Nx).real(fftw_out[i][1]);
                });
            }
            Kokkos::fence();

            fftw_destroy_plan(plan);
            fftw_free(fftw_in);
            fftw_free(fftw_out);
        }
        
        // GPU
        static void dfs_forwardCUFFT(ViewType data){
#ifdef __CUDACC__
            Int Nx = data.extent(0);
            Int Ny = data.extent(1);

            cufftHandle plan;
            cufftComplex *d_data;

            cudaMalloc ((void**)&d_data, sizeof(cufftComplex) * (Nx * Ny));
            if (data.layout() == Kokkos:: LayoutLeft){
                Kokkos::parallel_for(Nx * Ny, KOKKOS_LAMBDA(const Int i){
                    d_data[i].x = data(i / Ny, i % Ny).real();
                    d_data[i].y = data(i / Ny, i % Ny).imag();
                });
            }
            else{
                Kokkos::parallel_for(Nx * Ny, KOKKOS_LAMBDA(const Int i){
                    d_data[i].x = data(i % Nx, i / Nx).real();
                    d_data[i].y = data(i % Nx, i / Nx).imag();
                });
            }
            cudaDeviceSynchronize();

            cufftPlan2d(&plan, Nx, Ny, CUFFT_C2C);
            cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
            
            // transfer results to Kokkos Views
            if (data.layout() == Kokkos::LayoutLeft){
                Kokkos::parallel_for(Nx * Ny, KOKKOS_LAMBDA(const Int i){
                    data(i / Ny, i % Ny).real(d_data[i].x);
                    data(i / Ny, i % Ny).imag(d_data[i].y);
                });
            }
            else{
                Kokkos::parallel_for(Nx * Ny, KOKKOS_LAMBDA(const int i){
                    data(i % Nx, i / Nx).real(d_data[i].x);
                    data(i % Nx, i / Nx).imag(d_data[i].y);
                });
            }
            cudaDeviceSynchronize();

            cufftDestroy(plan);
            cudaFree(d_data);

else
            std::cerr << "Error: CUFFT requires CUDA support but not compiled with CUDA."<<std::endl;

#endif
        }
        

};

// Compute coefficients of a 2d array passed as 1d array
template<typename srcType, typename ViewType>
class DFS_FFTProcessor{
    private:
        Int nrows;
        Int ncols;
        view_2d<Real> dest_;

         // Function for copying 1D to 2D array
        void dfs_copy_to2d(SrcType src) const {
            if (src.layout() == Kokkos::LayoutRight){
                Kokos::parallel_for("Copy to 2D(LayoutRight)", nrows_, KOKKOS_LAMBDA(const Int i){
                    for (Int j=0; J <ncols_; ++j){
                        dest_(i,j) = src(i*ncols_ + j)
                    }
                });
            }
            else if (src.layout() == Kokkos::LayoutLeft)
            {
                Kokkos::Parallel_for("Copy to 2D (LayoutLeft)", nrows_, KOKKOS_LAMBDA(const Int i){
                    for (Int j=0; j < ncols_; ++j){
                        dest_(i,j) = src(j * nrows_ + i)
                    }
                });
            }
            else{
                // Unsupported layout
                Kokos::abort("unsupported layout for src array")
            }
        }

        /* 
        Apply the DFS glide reflection on a 2d array of samples on 
        colatitude -longitude points on the sphere
        */
       void dfs_doubling(ViewType utild){
           const Int nrows = utilde.extent(0) / 2 + 1;
           const Int ncols = utild.extent(1);
           const Int mid = utild.extent(1)/2;

           Kokkos::parallel_for("unshifted grid reflection", nrows-1, 
           KOKKOS_LAMBDA(const Int i){
            for(int j = 0; j < mid; ++j){
                utild(nrows-2-i, j) = dest_(i+1, j+mid);
                utild(nrows-2-i, j+mid) = dest_(i+1, j);
            }
           });

           Kokkos::parallel_for("unshifted grid reflection", nrows-1,
            KOKKOS_LAMBDA(const Int i){
                for(int j = 0; j < ncols; ++j)
                {
                    utild(nrows-1+i, j) = dest_(i, j);
                }
            }
           );
        }

        // scale Coefficients to be able to interpolate on the correct domain
        void dfs_interpCoeffs(ViewType data)
        {
            const Int mid = data.extent(0) / 2;
            Kokkos::parallel_for("shifts", data.extent(0), KOKKOS_LAMBDA(const Int i){
                Real cn = power(-1.0, -mid+i);
                for(Int j = 0; j< data.extent(1); ++j)
                {
                    data(i,j) = data(i,j) * cn / (data.extent(0)*data.extent(1)*1.0);
                }
            });
        }

    public:
        // constructor
        DFS_FFTProcessor(Int nrows, ncols): dest_("dest", nrows, ncols),nrows_(nrows),ncols_(ncols){}
        
        // Default destructor
        ~DFS_FFTProcessor(){} 

        // Fuction to compute FFTSHIFT
        void dfs_fftShift(ViewType data)
        {
            // Get the number of rows and columns
            Int nrows = data.extent(0);
            Int ncols = data.extent(1);
            
            // Mid point
            Int mid;

            // Shift rows
            mid = nrows / 2;
            Kokkos::parallel_for("shiftrows", Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                {0,0}, {mid, ncols}), KOKKOS_LAMBDA(const Int i, const Int j){
                    Complex tmp = data(i, j);
                    data(i, j) = data(i + mid, j);
                    data(i + mid, j) = tmp;
                });

            // shift columns
            mid = cols / 2;
            Kokkos::parallel_for("shiftcolumns", Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                {0, 0}, {nrows, mid}),
                KOKKOS_LAMBDA(const Int i, const Int j){
                    Complex tmp = data(i, j);
                    data(i, j) = data(i, j + mid);
                    data(i, j + mid) = tmp;
                }
            );
        }

        /*
         computes 2D Fourier coefficients of a function sampled
         on a latitude and longitude grid.
         Note: Create an instance of this class first
         before calling this function.
         */
        void dfs_FFT(srcType src, ViewType utild)
        {
            // copy function from 1D to 2D
            this -> dfs_copy_to2d(src);  
            // Double up function using glide reflection 
            this -> dfs_doubling(utild);
            // se FFTW or CUFFT to compute  FFT transform
            KokkosFFT<ViewType>::forwardFFT(utild);

            // Compute fftshift
            this -> dfs_fftShift(utild);

            // multiplying by (-1)^k to interpolate on [-pi,pi] time [0, 2*pi]
            this -> dfs_fftShift(utild);

        }

};

}
#endif