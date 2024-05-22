#include "dfs_rhs.hpp"

// Implement the constructor here
namespace SpherePoisson{
    // the constructor
    template<typename srcType, typename ViewType>
    DFS_RHS<srcType, ViewType>::DFS_RHS(const Int nrows_, const Int ncols_)
    {
        nrows = nrows_;
        ncols = ncols_;
        drhs("coefss rhs", ncols, ncols);
        sdrhs("coefss rhs", ncols, ncols);
        rhse("even coeffs", ncols/2, ncols);
        rhso("old coeffs", ncols/2, ncols);
        ie("even indices", ncols/2);
        io("old indices", ncols/2);

        // adjust FFT to the intended size
        FFTP(nrows, ncols);  
    }

    // private methods
    // update the splitting indices
    template<typename srcType, typename ViewType>
    void DFS_RHS<srcType, ViewType>::indices_split(){
        Int size = ncols/2;
        
        Kokkos::parallel_for(size, KOKKOS_LAMBDA(Int i){
            io(i) = 1 - (size % 2) + 2*i;
            ie(i) = size % 2  + 2*i;
        });
    }

    template<typename srcType, typename ViewType>
    void DFS_RHS<srcType, ViewType>::splitrhs()
    {
        Kokkos::parallel_for("split", Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0,0},{ncols/2, ncols}), KOKKOS_LAMBDA(const Int i, const Int j){
                rhso(i,j) = sdrhs(io(i), j);
                rhse(i,j) == sdrhs(ie(i), j);
            }
        );
    }

    template<typename srcType, typename ViewType>
    void DFS_RHS<srcType, ViewType>::scalerhs()
    {
        Kokkos::parallel_for("shiftrows",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {ncols, ncolss}), KOKKOS_LAMBDA (const int i, const int j) {
    
            double a = -0.125;
            double b = -0.25;
            double c = 0.5;
            if(i == 0){
                
                sdrhs(i,j) = c*drhs(i,j) + b*(drhs(i+2,j) +  drhs(nrows-2,j));
            }
            else if(i==1){
                sdrhs(i,j) = c*drhs(i,j) + b*drhs(i+2,j);
                
            }
            else if(i==nrows-2){
                sdrhs(i,j) = a*drhs(0,j) + b*drhs(i-2,j) + c*drhs(i,j);

            }
            else if(i==nrows-1){
                 sdrhs(i,j) =  b*drhs(i-2,j) + c*drhs(i,j);

            }
            else{
                 Real d =  i == 2 ? 1.0 : 2.0;
                sdrhs(i,j) = d*a*drhs(i-2,j) + c * drhs(i,j)+ b * drhs(i+2,j);
    

            }
        });

    }

    // The public methods
    /*
    compute the fft of the right handsides of the Poisson
    equation, scales it, and splits it to odd and even parts
    To get the we use public get methods.
    */ 
    template<typename srcType, typename ViewType>
    void DFS_RHS<srcType, ViewType>::poisson_rhs(srcType f)
    {
        FFTP.dfs_FFT(f, drhs);
        this->scalerhs();
        this->indices_split();
        this->splitrhs();
        
    }

    template<typename srcType, typename ViewType>
    view_1d<Int> DFS_RHS<srcType, ViewType>::get_io()
    {
        return io;
    }

   template<typename srcType, typename ViewType>
   view_1d<Int> DFS_RHS<srcType, ViewType>::get_ie()
   {
        return ie;
   }

   template<typename srcType, typename ViewType>
   ViewType DFS_RHS<srcType, ViewType>::get_rhse()
   {
        return rhse;
   }

    template<typename srcType, typename ViewType>
    ViewType DFS_RHS<srcType, ViewType>::get_rhso()
    {
        return rhso;
    }

}