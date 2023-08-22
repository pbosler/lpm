#ifndef DFS_LAPLACIAN_NEW_HPP
#define DFS_LAPLACIAN_NEW_HPP

#include "kokkos_dfs_types.hpp"
#include <KokkosBlas3_gemm.hpp>

namespace SpherePoisson {


    // We split the problem into even odd parts
    // which gives a speedup by a factor of 2
    // Lmato - odd part of the Laplacian matrix
    // Lmate - even part of the Laplacian matrix
    struct even_odd_split
    {
        view_2d<Real> Im;
        view_2d<Real> Io;
        view_2d<Real> Ie;
        view_2d<Complex> Lmat;
        view_2d<Complex> Lmate;
        view_2d<Complex> Lmato;
        view_1d<Int> ie;
        view_1d<Int> io;

        even_odd_split(view_2d<Real> Im_, view_2d<Real> Io_,  view_2d<Real> Ie_, view_2d<Complex> Lmat_,
        view_2d<Complex> Lmate_, view_2d<Complex> Lmato_, view_1d<Int> ie_, view_1d<Int> io_)
       :Im(Im_), Io(Io_), Ie(Ie_), Lmat(Lmat_), Lmate(Lmate_), Lmato(Lmato_), ie(ie_), io(io_) {}

        KOKKOS_INLINE_FUNCTION
        void operator()(const Int i, const Int j) const{
            // odd part
            Int n = Im.extent(0)/2;
            Int l = 1 - (n % 2) +  2*i;
            Int k = 1 - (n % 2) + 2*j;
            Lmato(i,j) = Lmat(l, k);
            Io(i,j) = Im(l, k);
            io(i) = l;

            // even part
            l = (n % 2) +  2*i;
            k =  (n % 2) + 2*j;
            Lmate(i,j) = Lmat(l, k);
            Ie(i,j) = Im(l, k);
            ie(i) = l;


        }
        
    };

    // Multiplying sin^2 the  and 
    //D2, the second derivative matrix Fourier space
    // and add to K^2sin^2
    // sin^2*D2 + K^2sin^2
    void multiply_sin_D2(view_2d<Real> Sin_d2,  Real Kappa);

    // This will compute
    // Mcos_sin_d1 = cos (t) * sin(t) * D1
    // in Fourier space, where D1 is
    // is the first derivative
    void multiply_cos_sin_d1(view_2d<Real> Mcos_sin_d1);

    // Computes the Laplacian matrix for differentiating
    // in Fourier space
    void laplacian_matrix(view_2d<Real> L_matrix, Real Kappa);

    // In the numerical solution we actually split L into even
    // and odd part to speed up the computation
    // Here I use the knowledge of structure of L to split it
    // even and odd part. The knowledge of the structure helps
    // so that only nonzeros are copied
    // Le_matrix := even part
    //Lo_matrix := odd part

    void even_odd_laplacian_matrix(view_2d<Real> Lo_matrix, view_2d<Real> Le_matrix, Real Kappa);

    
}
#endif