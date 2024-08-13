#include "dfs_rhs_new.hpp"



namespace SpherePoisson {
    void interp_shifts(const GridType grid_type, view_1d<Complex> cn)
    {
        Int mid = cn.extent(0)/2;
        
        if (grid_type ==  GridType::Shifted)
        {

            Kokkos::parallel_for("shifts",  cn.extent(0), KOKKOS_LAMBDA (const int i) {
                Int k =  -mid + i;
                cn[i] = exp(Complex(0.,1.0) *(-M_PI * k/double(cn.extent(0)))) * pow(-1.0, k);
               //cn(i) =  pow(-1.0, k);
            });

        }
        else if(grid_type == GridType::Unshifted)
        {
            Kokkos::parallel_for("shifts",  cn.extent(0), KOKKOS_LAMBDA (const int i) {
                Int k = -mid + i;
                
                cn(i) =  pow(-1.0, k);
                
            });

        }
    }

    // function fftshift
    void fftshift(view_2d<Complex> data)
    {
        Int nrows = data.extent(0);
        Int ncols = data.extent(1);
        
        // row shifts
        Int mid = nrows/2;
        Kokkos::parallel_for("shiftrows",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {mid, ncols}), KOKKOS_LAMBDA (const int i, const int j) {
            Complex tmp = data(i, j);
            data(i, j) = data(i+mid, j);
            data(i+mid, j) = tmp;
        });

        // column shifts
        mid = ncols/2;
        Kokkos::parallel_for("columnrows",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrows, mid}), KOKKOS_LAMBDA (const int i, const int j) {
            Complex tmp = data(i, j);
            data(i,j) = data(i, j+mid);
            data(i, j+mid) = tmp;
        });
    }


    // Values to Coefficients for function defined on the sphere
    void vals2CoeffsDbl(GridType grid_type, view_1d<Complex> cn, view_1d<Real> f, view_2d<Complex> F)
    {
        // double up the grid
        Int nrows = F.extent(0);
        Int ncols = F.extent(1);
        view_2d<Real> f_tilde("double f", nrows, ncols);
        dfs_doubling(f_tilde, f, grid_type);

        // compute fft using fftw3
        fftw_plan plan;
        int nthreads = 8;	//omp_get_max_threads();
        fftw_init_threads();
        fftw_plan_with_nthreads(nthreads);
        fftw_complex *X;
        X = fftw_alloc_complex(sizeof(fftw_complex) * (nrows * ncols));
        Kokkos::parallel_for("Copy X",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrows, ncols}), KOKKOS_LAMBDA (const int i, const int j) {
            X[i*ncols+j][0] = f_tilde(i,j);
            X[i*ncols+j][1] = 0.;
        });
        plan = fftw_plan_dft_2d(nrows, ncols, X, X, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute_dft(plan, X, X);
        fftw_destroy_plan(plan);

        // Put the coefficients back into
        Kokkos::parallel_for("Copy X",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrows, ncols}), KOKKOS_LAMBDA (const int i, const int j) {
           F(i,j) = Complex(X[i*ncols+j][0], X[i*ncols+j][1]);
  
        });
    	
	fftw_free(X);
	fftw_cleanup_threads();
        // perform the bivariate fftshift
        fftshift(F);

        // Shift the coefficients for interpolation
        // Put the coefficients back into
        Kokkos::parallel_for("Shift",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrows, ncols}), KOKKOS_LAMBDA (const int i, const int j) {
           F(i,j) = F(i,j) * cn(i) / (nrows * ncols * 1.0);
  
        });
    }

    // Function for scaling right handside of poisson equation
    void scalerhs(view_2d<Complex> rhs, view_2d<Complex> nrhs)
    {
        Int nrows = rhs.extent(0);
        
        Kokkos::parallel_for("shiftrows",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrows, nrows}), KOKKOS_LAMBDA (const int i, const int j) {
    
            double a = -0.125;
            double b = -0.25;
            double c = 0.5;
            if(i == 0){
                
                nrhs(i,j) = c*rhs(i,j) + b*(rhs(i+2,j) +  rhs(nrows-2,j));
            }
            else if(i==1){
                nrhs(i,j) = c*rhs(i,j) + b*rhs(i+2,j);
                
            }
            else if(i==nrows-2){
                nrhs(i,j) = a*rhs(0,j) + b*rhs(i-2,j) + c*rhs(i,j);

            }
            else if(i==nrows-1){
                 nrhs(i,j) =  b*rhs(i-2,j) + c*rhs(i,j);

            }
            else{
                 Real d =  i == 2 ? 1.0 : 2.0;
                nrhs(i,j) = d*a*rhs(i-2,j) + c * rhs(i,j)+ b * rhs(i+2,j);
    

            }
        });


     }

     // Functions for splitting into even old parts
      void indices_split(view_1d<int>io, view_1d<int>ie)
      {
        Int nrows = io.extent(0);
        Kokkos::parallel_for(nrows, [=](Int i){
            io(i) = 1 - (nrows % 2) + 2*i;
            ie(i) = nrows % 2 + 2*i;
        });
      }

    // Function for splitting the right handside of the
    // poisson equation into even and odd components
    void splitrhs(view_1d<int>ie, view_1d<int>io, view_2d<Complex> rhs, view_2d<Complex> rhso, view_2d<Complex> rhse)
    {
        Int nrows = ie.extent(0);
        Int ncols = rhs.extent(1);
        Kokkos::parallel_for("split",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrows, ncols}), KOKKOS_LAMBDA (const int i, const int j) {
        
          rhso(i, j) = rhs(io(i),j);
          rhse(i, j) = rhs(ie(i),j);
  
        });
    }

    // Function to prepare the right hand side of the poisson equation
    // for solving the equation in the Fourier coefficient space
     void poisson_rhs(GridType grid_type, view_1d<Int>ie, view_1d<Int>io, view_1d<Complex>cn, view_1d<Real> f, view_2d<Complex> rhso, view_2d<Complex> rhse)

    {
        const int nrows = 2* ie.extent(0);
        const int ncols = nrows;

        // Compute Fourier coefficients using DFS
        view_2d<Complex> rhs("coeffs",nrows, ncols);
        view_2d<Complex> nrhs("coeffs",nrows, ncols);
       
        vals2CoeffsDbl(grid_type, cn, f, rhs);

        // Scale the right handside by sin^2
        scalerhs(rhs, nrhs);

        
        // Compute the even and odd split
       splitrhs(ie, io, nrhs, rhso, rhse);

        
    }

    // Function for computing value given Fourier coefficients
    // on the sphere.
    void coeffs2valsDbl(GridType grid_type, view_1d<Complex> cn, view_2d<Complex> F, view_2d<Real> f)
    {
        Int dnrows = F.extent(0);
        Int ncols = F.extent(1);
        Int nrows = dnrows / 2;

        // Shift back using fftshift
        Kokkos::parallel_for("Shift",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {dnrows, ncols}), KOKKOS_LAMBDA (const int i, const int j) {
           F(i,j) = F(i,j) * cn(i);
  
        });

        fftshift(F);

        // 2D FFT
        fftw_complex *X;
        fftw_plan plan;
        Int nthreads = 8; // omp_get_max_threads();
        fftw_init_threads();
        fftw_plan_with_nthreads(nthreads);
        X = fftw_alloc_complex(sizeof(fftw_complex) * (dnrows * ncols));
        Kokkos::parallel_for("Copy X",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {dnrows, ncols}), KOKKOS_LAMBDA (const int i, const int j) {
           X[i*ncols+j][0] = F(i,j).real();
           X[i*ncols+j][1] = F(i, j).imag();
        });

        plan = fftw_plan_dft_2d(dnrows, ncols, X, X, FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_execute_dft(plan, X, X);
        fftw_destroy_plan(plan);

        // Relate back to the original grid
        Int w = ncols / 2;
        if(grid_type == GridType::Shifted)
        {
            Kokkos::parallel_for("transback",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrows, ncols}), KOKKOS_LAMBDA (const int i, const int j) {
            
              f(i,j) = X[(nrows+i)*ncols+j][0];
             
        });
            
        }
        else if(grid_type == GridType::Unshifted)
        {
            Kokkos::parallel_for("transback",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrows, ncols}), KOKKOS_LAMBDA (const int i, const int j) {
                f(i,j) = X[(nrows+i)*ncols+j][0];
            });
            
            // copy the final row
            Kokkos::parallel_for("transback", ncols/2, KOKKOS_LAMBDA (const int j) {
                f(nrows,j) = X[j+w][0];
                f(nrows,j+w) = X[j][0];
            });
        }
	
	fftw_free(X);
	fftw_cleanup_threads();
    
    }


}
