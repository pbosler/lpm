#include "dfs_solve_new.hpp"

namespace SpherePoisson {

     // Thomas algorithm: the tridiagonal solver
     // In our application we expect real RHS and
     template<typename T>
     void tridiag_solver(view_2d<T> A, view_2d<Complex> d, Int k)
     {
        // forward propagation
        Int n = A.extent(0);
        if(abs(A(n-1,0)) + abs(A(0, n-1)) > 0)  // if true use the modified Thomas algorithm
        {
            for(int i=0; i<n-2; i++)
            {
                T tmp = -A(n-1,i)/A(i,i);
                A(n-1,n-1) = tmp * A(i, n-1) + A(n-1, n-1);
                A(n-1, i+1)  = tmp * A(i, i+1) + A(n-1, i+1);
                d(n-1,k) = tmp*d(i,k) + d(n-1,k);
                A(n-1,i)=0;

                tmp = -A(i+1, i)/A(i,i);
                A(i+1, i+1) = tmp * A(i, i+1) + A(i+1, i+1);
                d(i+1,k) = tmp * d(i,k) + d(i+1,k);
                A(i+1, i) = 0;
                A(i+1, n-1) = tmp * A(i, n-1) + A(i+1, n-1);
            }
        
            Int jj= n-2;
            T tmp = - A(n-1, jj) / A(jj, jj);
            A(n-1, jj+1) = tmp*A(jj, jj+1) + A(n-1, jj+1);
            d(n-1,k) = tmp*d(jj,k) + d(n-1,k);
            A(n-1, jj) = 0;

            d(n-1,k) = d(n-1,k)/A(n-1, n-1);
            d(n-2,k) = (d(n-2,k) - A(n-2,n-1)*d(n-1,k))/A(n-2, n-2);
            for(int i=n-3; i>-1; i--)
            {
                d(i,k) = (d(i,k) - A(i, i+1)*d(i+1,k) -A(i, n-1)*d(n-1,k))/A(i,i);
            }
        }
        else    // use the normal Thomas algorithm
        {
             T w;
             // forward substitution
             for(int i=1; i<n; i++)
             {
                w = A(i, i-1)/A(i-1,i-1);
                A(i,i) = A(i,i) - w * A(i-1,i);
                d(i,k) = d(i,k) - w*d(i-1,k);
             }

            // backward substitution
            d(n-1,k) = d(n-1,k) / A(n-1, n-1);
            for(int i=n-2; i>-1; i--)
            {
                d(i,k) = (d(i,k) - A(i,i+1)*d(i+1,k))/A(i,i);
            }
        }
     }
template void tridiag_solver<Complex>(view_2d<Complex> AA, view_2d<Complex> name, Int k);
template void tridiag_solver<Real>(view_2d<Real> AA, view_2d<Complex> name, Int k);

// Triadiagonal solver 3d
template<typename T>
void tridiag_solver_3d(view_3d<T> A, view_2d<Complex> d, Int k)
{
    // forward propagation
    Int n = A.extent(0);
    if(abs(A(n-1,0, k)) + abs(A(0, n-1, k)) > 0)  // if true use the modified Thomas algorithm
    {
        for(int i=0; i<n-2; i++)
        {
            T tmp = -A(n-1,i, k)/A(i,i, k);
            A(n-1,n-1, k) = tmp * A(i, n-1, k) + A(n-1, n-1, k);
            A(n-1, i+1, k)  = tmp * A(i, i+1, k) + A(n-1, i+1, k);
            d(n-1,k) = tmp*d(i,k) + d(n-1,k);
            A(n-1,i, k)=0;

            tmp = -A(i+1, i, k)/A(i,i, k);
            A(i+1, i+1, k) = tmp * A(i, i+1, k) + A(i+1, i+1, k);
            d(i+1,k) = tmp * d(i,k) + d(i+1,k);
            A(i+1, i, k) = 0;
            A(i+1, n-1, k) = tmp * A(i, n-1, k) + A(i+1, n-1, k);
        }
        
        Int jj= n-2;
        T tmp = - A(n-1, jj, k) / A(jj, jj, k);
        A(n-1, jj+1, k) = tmp*A(jj, jj+1, k) + A(n-1, jj+1, k);
        d(n-1,k) = tmp*d(jj,k) + d(n-1,k);
        A(n-1, jj, k) = 0;

            d(n-1,k) = d(n-1,k)/A(n-1, n-1,k);
            d(n-2,k) = (d(n-2,k) - A(n-2,n-1,k)*d(n-1,k))/A(n-2, n-2,k);
            for(int i=n-3; i>-1; i--)
            {
                d(i,k) = (d(i,k) - A(i, i+1,k)*d(i+1,k) -A(i, n-1,k)*d(n-1,k))/A(i,i,k);
            }
    }
    else    // use the normal Thomas algorithm
    {
        T w;
        // forward substitution
        for(int i=1; i<n; i++)
        {
            w = A(i, i-1,k)/A(i-1,i-1,k);
            A(i,i,k) = A(i,i,k) - w * A(i-1,i,k);
            d(i,k) = d(i,k) - w*d(i-1,k);
        }

        // backward substitution
        d(n-1,k) = d(n-1,k) / A(n-1, n-1,k);
        for(int i=n-2; i>-1; i--)
        {
            d(i,k) = (d(i,k) - A(i,i+1,k)*d(i+1,k))/A(i,i,k);
        }
    }
}
template void tridiag_solver_3d<Complex>(view_3d<Complex> AA, view_2d<Complex> name, Int k);
template void tridiag_solver_3d<Real>(view_3d<Real> AA, view_2d<Complex> name, Int k);



 // Prepare matrix Le  for the zero mode when singular
  void rearrange(view_2d<Complex> Le, view_1d<Complex> fe, view_1d<Int> ie)
  {
    Int n = Le.extent(0);
    Int m = 2*n;
    Int mid = n/2;

    // Exachange last row with mid row
    // Exchange mid column with last column
    // Here only target indices with nonzero values
    Le(0, mid) = Le(0, n-1);    Le(0, n-1) = Complex(0.0);
    Le(mid, 0) = Le(n-1, 0);
    Le(mid, mid) = Le(n-1, n-1);
    Le(mid, n-2) = Le(n-1, n-2);
    Le(n-2, mid) = Le(n-2, n-1); Le(n-2, n-1) = Complex(0.0);

    Le(mid, mid-1)=0; Le(mid, mid+1) = 0;

    // Exchange the rows of the right handside aswell
    fe(mid) = fe(n-1);
    fe(n-1) = Complex(0.0);

    // Fix the singularity by updating the last row as follows
    view_1d<Complex> bc("constrain", m);
    Real ds = -double(n);
    Kokkos::parallel_for("solves", m, KOKKOS_LAMBDA(const Int i){
            Real mm = ds + i;
            bc(i) = 0.5 * (1.0 + exp(Complex(0., 1.) * M_PI * mm)) / (1 - mm*mm);
        
        });
        bc[n - 1] = Complex(0.0);
        bc[n + 1] = Complex(0.0);

        Kokkos::parallel_for("solves", n, KOKKOS_LAMBDA(const Int i){
            Le(n-1, i) = bc(ie(i));
        });
        
        // 
        Complex temp = Le(n-1, mid);
        Le(n-1,mid) = Le(n-1, n-1);
        Le(n-1,n-1) = temp;
  }

  // special solver
   void special_solver(view_2d<Complex> Le, view_1d<Complex> fe, view_1d<Int> ie)
   {
        Complex tmp;
        Int n = Le.extent(0);
        Int mid = int(n/2)+1;
        
        // Fix the singularity problem
        rearrange(Le, fe, ie);

        // Solve Le * x = fe 
        // Forward propagation
        // First half of the pivots
        for(int i = 0; i < mid-2; i++)
        {
            // bottom row
            tmp = -Le(n-1,i) / Le(i,i);
            Le(n-1, mid-1) = tmp*Le(i, mid-1) + Le(n-1, mid-1);

            Le(n-1, i+1) = tmp*Le(i, i+1) + Le(n-1, i+1);
            fe(n-1) = fe(n-1) + tmp*fe(i);
            Le(n-1,i) = 0;

            // mid row
            tmp = - Le(mid-1, i) / Le(i, i);
            Le(mid-1, i+1) = tmp*Le(i, i+1) + Le(mid-1, i+1);
            fe(mid-1) = fe(mid-1) + tmp*fe(i);
            Le(mid-1, i)  = 0;
            Le(mid-1,mid-1) = tmp*Le(i,mid-1) + Le(mid-1, mid-1);

            // immediate next row
            tmp = -Le(i+1, i)/Le(i,i);
            Le(i+1,i+1) = tmp*Le(i,i+1) + Le(i+1,i+1);
            fe(i+1) = tmp*fe(i) + fe(i+1);
            Le(i+1,i) = 0;
            Le(i+1, mid-1) = tmp*Le(i,mid-1) + Le(i+1, mid-1);
        }
        
        // structure of matrix demands special treatment at i=mid-2
        int jj = mid-2;
        tmp = - Le(n-1, jj)/Le(jj, jj);
        Le(n-1, jj+1) = tmp*Le(jj, jj+1) + Le(n-1, jj+1);
        fe(n-1) = tmp*fe(jj) + fe(n-1);
        Le(n-1, jj)=0;

        tmp = -Le(jj+1, jj)/Le(jj, jj);
        Le(jj+1, jj+1) = tmp*Le(jj, jj+1) + Le(jj+1, jj+1);
        fe(jj+1) = tmp*fe(jj) + fe(jj+1);
        Le(jj+1, jj) = 0;

        // forward propagation through remaining half
        for(int i=mid-1; i<n-2; i++)
        {
            if(i < n-3)
            {
                tmp = -Le(n-1, i)/ Le(i, i);
                Le(n-1,n-2) = tmp*Le(i,n-2) +  Le(n-1,n-2);
                Le(n-1,i+1) = tmp*Le(i,i+1) +  Le(n-1,i+1);
                fe(n-1) = tmp*fe(i) + fe(n-1);
                Le(n-1,i)=0;

                tmp = - Le(n-2, i)/Le(i,i);
                Le(n-2,n-2) = tmp*Le(i,n-2) +  Le(n-2,n-2);
                Le(n-2,i+1) = tmp*Le(i,i+1) +  Le(n-2,i+1);
                fe(n-2) = tmp*fe(i) + fe(n-2);
                Le(n-2,i)=0;

                tmp = - Le(i+1, i)/Le(i,i);
                Le(i+1,i+1) = tmp*Le(i,i+1) +  Le(i+1,i+1);
                fe(i+1) =  tmp*fe(i) + fe(i+1);
                Le(i+1,i)=0;
            }
            else 
            {
                tmp = -Le(n-1,i)/Le(i,i);
                Le(n-1,i+1) = tmp*Le(i,i+1) + Le(n-1,i+1);
                fe(n-1) =  tmp*fe(i) + fe(n-1);
                Le(n-1,i)=0;
                
                tmp = -Le(n-2,i)/Le(i,i);
                Le(n-2,i+1) = tmp*Le(i,i+1) +  Le(n-2,i+1);
                fe(n-2) = tmp*fe(i) + fe(n-2);
                Le(n-2,i)=0;
            }
        }

        // Special treatment at pivot = Le(n-2, n-2);
        jj = n - 2;
        tmp = -Le(n-1, jj)/Le(jj, jj);
        Le(n-1, jj+1) = tmp*Le(jj, jj+1) + Le(n-1, jj+1);
        fe(n-1) = tmp*fe(jj) + fe(n-1);
        Le(n-1, jj) = 0;

        // Backward propagation
        fe(n-1) = fe(n-1) / Le(n-1, n-1);
        fe(n-2) = fe(n-2) / Le(n-2, n-2);

        for(int i = n-3; i > mid-1; i--)
        {
            fe(i) = (fe(i) - Le(i, i+1)*fe(i+1)) / Le(i, i);
        }

        fe(mid - 1) = (fe(mid-1) - Le(mid-1, n-2)*fe(n-2)) / Le(mid-1, mid-1);
        fe(mid - 2) = (fe(mid-2) - Le(mid-2, mid-1)*fe(mid-1)) / Le(mid-2, mid-2);

        for(int i=mid-3; i>-1; i--)
        {
            fe(i) = (fe(i) - Le(i, i+1)*fe(i+1) - Le(i, mid-1)*fe(mid-1)) / Le(i, i);

        }

        // exchange the fe(n-1) and  fe(mid-1) again
        tmp = fe(n-1);
        fe(n-1) = fe(mid-1);
        fe(mid-1) = tmp;
   }
     

  // the full solver   
     void solver(view_2d<Real> Le, view_2d<Real> Lo, view_2d<Complex> rhse,  view_2d<Complex> rhso, view_2d<Complex> UU,
    view_1d<Int> io, view_1d<Int> ie, Real Kappa)
     {
	/* To solve the system LU=F we split both L and F into even 
	 * odd mode parts. Here rhse and rhso are for even and odd
	 * modes of F, respectively. Where as Lle and Llo are the even
	 * modes of L. This decoupling of the equation reduces the
	 * computation cost of the method by a factor of two.
	 */
        Int ncols = rhse.extent(1);
        Int nrows = rhse.extent(0);
        Int mid = ncols/2;

        view_3d<Real> Llo("Lok", nrows, nrows, mid);
        view_3d<Complex> Lle("Lek", nrows, nrows, mid);
        view_1d<Real> dd("shifts", mid);
	
	LPM_ASSERT_MSG( (UU.extent(0) == 2*nrows and UU.extent(1) == ncols), "UU size error");

        Kokkos::parallel_for(mid, [=](Int k){
            dd(k) =- pow(double(mid - k),2);
            for(int i=0; i<nrows; i++)
            {
                for(int j=0; j<nrows; j++)
                {
                    Llo (i,j,k) = Lo(i,j);
                    Lle(i,j,k) = Le(i,j);

                }
                Llo(i,i,k) = Llo(i,i,k) + dd(k);
                Lle(i,i,k) = Lle(i,i,k) + dd(k);
            }
            
            tridiag_solver_3d(Lle, rhse, k);  
            tridiag_solver_3d(Llo, rhso, k);
  
            // copy to the output array
            for(int i=0; i<nrows; i++)
            { 
                UU(io(i),k) = rhso(i, k);
                UU(ie(i),k) = rhse(i, k);
            }
        });
        
        // Solving for the odd zero mode
        view_2d<Real> Lo_k("Lok", nrows, nrows);
        view_2d<Complex> Le_k("Lek", nrows, nrows);
        view_1d<Complex> fel("fe", nrows);
        Kokkos::parallel_for(nrows, [=](Int i){
            if(i !=0){
                Lo_k(i,i-1) = Lo(i,i-1);
                Le_k(i,i-1) = Le(i,i-1);
            }
            if (i != nrows -1){
                Lo_k(i,i+1) = Lo(i,i+1);
                Le_k(i,i+1) = Le(i,i+1);
            }
            
            Lo_k(i,i) = Lo(i,i); 
            Le_k(i,i) = Le(i,i);

            fel(i) = rhse(i,mid);

     });

    Lo_k(0,nrows-1) = Lo(0,nrows-1);
    Lo_k(nrows-1,0) = Lo(nrows-1,0);
    tridiag_solver(Lo_k, rhso, mid);
    
    Le_k(0,nrows-1) = Le(0,nrows-1);
    Le_k(nrows-1,0) = Le(nrows-1,0);

      
     if(Kappa == 0 )
    {
        // Use the integral constraint to regularize
        special_solver(Le_k, fel, ie);
       
     }
     else
     {
        tridiag_solver(Le_k, rhse, mid);
     }
    
    // Update UU for the zero mode
    Kokkos::parallel_for("solves", nrows, KOKKOS_LAMBDA(const Int i){
            UU(io(i), mid) = rhso(i,mid);
            if (Kappa == 0)
            {
            UU(ie(i), mid) = fel(i);
            }
            else
            {
                UU(ie(i), mid) = rhse(i, mid);
            }
        
        });


    // Filling the positive modes from the negative nodes
 
    Real tpm = pow(-1.0, mid);
    Kokkos::parallel_for("copyU",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrows, mid-1}), KOKKOS_LAMBDA (const int i, const int j) {
        Int k_neg = mid-1-j;
        Int jj = mid + 1 + j;
        UU(io(i), jj ) =  tpm * pow(-1.0, jj + 1) * (-conj(UU(io(i),k_neg)));
        UU(ie(i), jj) = tpm * pow(-1.0, jj + 1) * (-conj(UU(ie(i),k_neg)));
    
    });

}


}
