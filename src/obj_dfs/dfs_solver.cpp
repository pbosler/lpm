#include "dfs_solver.hpp"

namespace SpherePoisson
{
    template<typename srcType, typename ViewType>
    void DFSSolver<srcType, ViewType>::tridiag_solver(view_2d<Real> A, ViewType D, Int k )
    {   // forward propagation
        Int n = A.extent(0);
        if(abs(A(n-1,0)) + abs(A(0, n-1)) > 0)  // if true use the modified Thomas algorithm
        {
            for(int i=0; i<n-2; i++)
            {
                Real tmp = -A(n-1,i)/A(i,i);
                A(n-1,n-1) = tmp * A(i, n-1) + A(n-1, n-1);
                A(n-1, i+1)  = tmp * A(i, i+1) + A(n-1, i+1);
                D(n-1,k) = tmp*D(i,k) + D(n-1,k);
                A(n-1,i)=0;

                tmp = -A(i+1, i)/A(i,i);
                A(i+1, i+1) = tmp * A(i, i+1) + A(i+1, i+1);
                D(i+1,k) = tmp * D(i,k) + D(i+1,k);
                A(i+1, i) = 0;
                A(i+1, n-1) = tmp * A(i, n-1) + A(i+1, n-1);
            }
        
            Int jj= n-2;
            Real tmp = - A(n-1, jj) / A(jj, jj);
            A(n-1, jj+1) = tmp*A(jj, jj+1) + A(n-1, jj+1);
            D(n-1,k) = tmp*D(jj,k) + D(n-1,k);
            A(n-1, jj) = 0;

            D(n-1,k) = D(n-1,k)/A(n-1, n-1);
            D(n-2,k) = (D(n-2,k) - A(n-2,n-1)*D(n-1,k))/A(n-2, n-2);
            for(int i=n-3; i>-1; i--)
            {
                D(i,k) = (D(i,k) - A(i, i+1)*D(i+1,k) -A(i, n-1)*D(n-1,k))/A(i,i);
            }
        }
        else    // use the normal Thomas algorithm
        {
             Real w;
             // forward substitution
             for(int i=1; i<n; i++)
             {
                Real = A(i, i-1)/A(i-1,i-1);
                A(i,i) = A(i,i) - w * A(i-1,i);
                D(i,k) = D(i,k) - w*D(i-1,k);
             }

            // backward substitution
            D(n-1,k) = D(n-1,k) / A(n-1, n-1);
            for(int i=n-2; i>-1; i--)
            {
                D(i,k) = (D(i,k) - A(i,i+1)*D(i+1,k))/A(i,i);
            }
        }
    }

    //
    template<typename srcType, typename ViewType>
    void DFSSolver<srcType, ViewType>::tridiag_solver3d(view_3d<Real> A, ViewType D, Int k)
    {
        // forward propagation
    Int n = A.extent(0);
    if(abs(A(n-1,0, k)) + abs(A(0, n-1, k)) > 0)  // if true use the modified Thomas algorithm
    {
        for(int i=0; i<n-2; i++)
        {
            Real tmp = -A(n-1,i, k)/A(i,i, k);
            A(n-1,n-1, k) = tmp * A(i, n-1, k) + A(n-1, n-1, k);
            A(n-1, i+1, k)  = tmp * A(i, i+1, k) + A(n-1, i+1, k);
            D(n-1,k) = tmp*D(i,k) + D(n-1,k);
            A(n-1,i, k)=0;

            tmp = -A(i+1, i, k)/A(i,i, k);
            A(i+1, i+1, k) = tmp * A(i, i+1, k) + A(i+1, i+1, k);
            D(i+1,k) = tmp * D(i,k) + D(i+1,k);
            A(i+1, i, k) = 0;
            A(i+1, n-1, k) = tmp * A(i, n-1, k) + A(i+1, n-1, k);
        }
        
        Int jj= n-2;
        Real tmp = - A(n-1, jj, k) / A(jj, jj, k);
        A(n-1, jj+1, k) = tmp*A(jj, jj+1, k) + A(n-1, jj+1, k);
        D(n-1,k) = tmp*D(jj,k) + D(n-1,k);
        A(n-1, jj, k) = 0;

            D(n-1,k) = D(n-1,k)/A(n-1, n-1,k);
            D(n-2,k) = (D(n-2,k) - A(n-2,n-1,k)*D(n-1,k))/A(n-2, n-2,k);
            for(int i=n-3; i>-1; i--)
            {
                D(i,k) = (D(i,k) - A(i, i+1,k)*D(i+1,k) -A(i, n-1,k)*D(n-1,k))/A(i,i,k);
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
            D(i,k) = D(i,k) - w*D(i-1,k);
        }

        // backward substitution
        D(n-1,k) = D(n-1,k) / A(n-1, n-1,k);
        for(int i=n-2; i>-1; i--)
        {
            D(i,k) = (D(i,k) - A(i,i+1,k)*D(i+1,k))/A(i,i,k);
        }
    }    
    }

    template<typename srcType, typename ViewType>
    void DFSSolver<srcType, ViewType>::rearrange(view_2d<Complex> Le, view_1d<Complex> fe, view_1d<Int>)
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

    template<typename srcType,typename ViewType>
    void DFSSolver<srcType,ViewType>::special_solver(view_2d<Complex> Le, view_1d<Complex> fe, view_1d<Int> ie)
    {
        Complex tmp;
        Int n = Le.extent(0);
        Int mid = int(n/2)+1;
        
        // Fix the singularity problem
        this->rearrange(Le, fe, ie);

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
    }

    // the public method
    template <typename srcType, typename ViewType>
     void DFSSolver<srcType, ViewType>::solver(LHS_Poisson& lhs_obj, DFS_RHS<srcType,ViewType> & rhs_obj, view_2d<Complex>UU)
     {
        Int n = lhs_obj.Lo_matrix.extent(0);
        Int ncols = 2*n;

        // make copies of the private data in rhs_obj
        ViewType rhse("even", n, ncols);
        ViewType rhso("odd", n, ncols);
        view_1d<Int> ie("evenindex",n);
        view_1d<Int> io("evenindex",n);

        rhse = rhs_obj.get_rhse();
        rhso = rhs_obj.get_rhso();
        ie = rhs_obj.get_ie();
        io = rhs_obj.get_io();

        view_3d<Real> Llo("Lok", n, n, n);
        view_3d<Real> Lle("Lek", n, n, n);
        view_1d<Real> dd("shifts", n);

        Kokkos::parallel_for(n, [=](Int k){
            dd(k) =- pow(double(n - k),2);
            for(int i=0; i<n; i++)
            {
                for(int j=0; j<n; j++)
                {
                    Llo (i,j,k) = lhs_obj.Lo_matrix(i,j);
                    Lle(i,j,k) = lhs_obj.Le_matrix(i,j);

                }
                Llo(i,i,k) = Llo(i,i,k) + dd(k);
                Lle(i,i,k) = Lle(i,i,k) + dd(k);
            }
            
            this -> tridiag_solver_3d(Lle, rhse, k);  
            this -> tridiag_solver_3d(Llo, rhso, k);
  
            // copy to the output array
            for(int i=0; i<n; i++)
            { 
                UU(io(i),k) = rhso(i, k);
                UU(ie(i),k) = rhse(i, k);
            }
        });

        // zeroth mode
        Real Kappa = lhs_obj.Kappa;
        if(Kappa != 0)
        {
            view_2d<Real> Lo_k("Lok", n, n);
            view_2d<Real> Le_k("Lek", n, n);
            
            Kokkos::parallel_for(n, [=](Int i){
            if(i !=0){
                Lo_k(i,i-1) = Lo(i,i-1);
                Le_k(i,i-1) = Le(i,i-1);
            }
            if (i != n -1){
                Lo_k(i,i+1) = Lo(i,i+1);
                Le_k(i,i+1) = Le(i,i+1);
            }
            
            Lo_k(i,i) = Lo(i,i); 
            Le_k(i,i) = Le(i,i);

            });

            Lo_k(0,n-1) = Lo(0,n-1);
            Lo_k(n-1,0) = Lo(n-1,0);
            Le_k(0,n-1) = Le(0,n-1);
            Le_k(n-1,0) = Le(n-1,0);
            this -> tridiag_solver(Lo_k, rhso, n);
            this -> tridiag_solver(Le_k, rhse, n);
        }
        else{
            view_2d<Real> Lo_k("Lok", n, n);
            view_2d<Complex> Le_k("Lek", n, n);
            view_1d<Complex> fe("rhse_k", n);
            
            Kokkos::parallel_for(n, [=](Int i){
            if(i !=0){
                Lo_k(i,i-1) = Lo(i,i-1);
                Le_k(i,i-1) = Le(i,i-1);
            }
            if (i != n -1){
                Lo_k(i,i+1) = Lo(i,i+1);
                Le_k(i,i+1) = Le(i,i+1);
            }
            
            Lo_k(i,i) = Lo(i,i); 
            Le_k(i,i) = Le(i,i);
            fe(i) = rhse(i,n);

            });

            Lo_k(0,n-1) = Lo(0,n-1);
            Lo_k(n-1,0) = Lo(n-1,0);
            Le_k(0,n-1) = Le(0,n-1);
            Le_k(n-1,0) = Le(n-1,0);
            this -> tridiag_solver(Lo_k, rhso, n);
            // Use the integral constraint to regularize
            this -> special_solver(Le_k, fe, ie);
        }

        // Update UU for the zero mode
        Kokkos::parallel_for("solves", nrows, KOKKOS_LAMBDA(const Int i){
            UU(io(i), n) = rhso(i,n);
            if (Kappa == 0)
            {
            UU(ie(i), n) = fe(i);
            }
            else
            {
                UU(ie(i), n) = rhse(i, n);
            }
        });

        // Filling the positive modes from the negative nodes
 
        Real tpm = pow(-1.0, n);
        Kokkos::parallel_for("copyU",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {n, n-1}), KOKKOS_LAMBDA (const int i, const int j) {
            Int k_neg = n-1-j;
            Int jj = n + 1 + j;
            UU(io(i), jj ) =  tpm * pow(-1.0, jj + 1) * (-conj(UU(io(i),k_neg)));
            UU(ie(i), jj) = tpm * pow(-1.0, jj + 1) * (-conj(UU(ie(i),k_neg)));
        });
     }
}