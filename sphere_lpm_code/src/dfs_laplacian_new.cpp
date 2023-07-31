#include "dfs_laplacian_new.hpp"

namespace SpherePoisson {

    // sin^2(theta) * D2 + K^2 * sin^2(theta)
    void multiply_sin_D2(view_2d<Real> Sin_d2,  Real Kappa)
    {
        Int m = Sin_d2.extent(0);
    
        // For Poisson K is zero
        if(Kappa ==0)
        {
            
            Kokkos::parallel_for(m, [=](Int i){
    
            Real a = -0.125;
            Real b = -0.25;
            Real c = 0.5;
            Real dd = double(-m/2.0);
            
            if(i==0)
            {
                Sin_d2(i,i) = -c * pow(dd,2);
                Sin_d2(i, i+2) = -b * pow((dd + (i+2)),2);
                Sin_d2(i, m-2) = -b * pow((dd + (m-2)),2);
            }
            else if(i==1)
            {
                Sin_d2(i,i) = -c * pow((dd + i),2);
                Sin_d2(i, i+2) = -b * pow((dd + i + 2),2);
            }
            else if(i==m-2)
            {
                Sin_d2(i,i) = -c * pow((dd + i),2);
                Sin_d2(i, i-2) = -b * pow((dd + i - 2),2);
                Sin_d2(i, 0) = -a * pow(dd,2);

            }
            else if(i==m-1)
            {
                Sin_d2(i, i-2) = -b * pow((dd + i - 2),2);
                Sin_d2(i,i) = -c * pow((dd + i),2);
            }
            else
            {
                Real d =  i == 2 ? 1.0 : 2.0;
                Sin_d2(i,i) = -c * pow(dd + i,2);
                Sin_d2(i,i-2) = -d * a * pow((dd + i - 2),2);
                Sin_d2(i, i+2) = -b * pow((dd + i + 2),2);

            }

            });

        
        }
        else
        {
            Kokkos::parallel_for(m, [=](Int i){
    
            Real a = -0.125;
            Real b = -0.25;
            Real c = 0.5;
            Real dd = double(-m/2);
            Real k2 = Kappa * Kappa;

            if(i==0)
            {
                Sin_d2(i,i) = c * (-pow(dd,2) + k2);
                Sin_d2(i, i+2) = b * (-pow(dd + (i+2),2) + k2);
                Sin_d2(i, m-2) = b * (-pow(dd + (m-2),2) + k2);
            }
            else if(i==1)
            {
                Sin_d2(i,i) = c * (-pow(dd + i,2) + k2);
                Sin_d2(i, i+2) = b * (-pow(dd + i + 2,2) + k2);
            }
            else if(i==m-2)
            {
                Sin_d2(i,i) = c * (-pow(dd + i,2) + k2);
                Sin_d2(i, i-2) = b * (-pow(dd + i - 2,2) + k2);
                Sin_d2(i, 0) = a * (-pow(dd,2) + k2);

            }
            else if(i==m-1)
            {
                Sin_d2(i, i-2) = b * (-pow(dd + i - 2,2) + k2);
                Sin_d2(i,i) = c * (-pow(dd + i,2) + k2);
            }
            else
            {
                Real d =  i == 2 ? 1.0 : 2.0;
                Sin_d2(i,i) = c * (-pow(dd + i,2) + k2);
                Sin_d2(i,i-2) = d * a * (-pow(dd + i - 2,2) + k2);
                Sin_d2(i, i+2) = b * (-pow(dd + i + 2,2) + k2);

            }

            });

        }

    }

    // Mcos_sin_d1 = cos(t)*sin(t)*D1
    // Mcos_sin_d1 is n by n matrix
    // We also assume n > = 6
    void multiply_cos_sin_d1(view_2d<Real> Mcos_sin_d1)
    {
        Int m = Mcos_sin_d1.extent(0);
        Int n = Mcos_sin_d1.extent(1);
        if((m != n) || (m < 6))
        {
            std::cerr<<"Error: Mcos_sin is either not a square matrix or nrows < 6"<<std::endl;
            exit(1);
        }

        Real dd =-m/2.0;


        Kokkos::parallel_for(m-4, [=](Int i){
            Real a = 0.25;
            Real b = -0.25;
            if( i==0)
            {
                Mcos_sin_d1(i, 2) = b * (dd + 2.0);
                Mcos_sin_d1(i, m-2) = a * (dd + m-2.0);
                Mcos_sin_d1(1, 3) = b * (dd + 3);
                Mcos_sin_d1(2, 4) = b * (dd + 4);
                Mcos_sin_d1(m-2, m-4) = a * (dd + m - 4);
                Mcos_sin_d1(m-1, m-3) = a * (dd + m - 3);

            }
            else{
                int l = 2+i;
                Mcos_sin_d1(l, l-2) = a * (dd + l -2);
                Mcos_sin_d1(l, l+2) = b * (dd + l + 2);

            }
        

        }
        );
    }

    // Computes the Laplacian matrix on the sphere
    // the matrix is expected to be a square matrix
    // atleast a 6 by 6 matrix
    void laplacian_matrix(view_2d<Real> L_matrix, Real Kappa)
    {
        Int m = L_matrix.extent(0);
        Int n = L_matrix.extent(1);

        if((m != n) || (m < 6))
        {
            std::cerr<<"Error: L_matrix is either not a square matrix or nrows < 6"<<std::endl;
            exit(1);
        }

        // Compute the sin^2 (theta) * D2 + K^2 * sin^2 (theta)
        // and save it  in L_matrix 
        multiply_sin_D2(L_matrix,  Kappa);

        // Compute cos(theta) * sin(theta)  * D1
        // save the result in a temporary matrix
        view_2d<Real> Mcos_sin_d1("McosSinD1", m, m);
        multiply_cos_sin_d1(Mcos_sin_d1);

        // Exploit the  structure of Mcos_sin_d1 to update 
        // L_matrix with values in Mcos_sin_d1
        Kokkos::parallel_for(m, [=](Int i){
            if(i == 0)
            {
                L_matrix(i, i+2) += Mcos_sin_d1(i, i+2);
                L_matrix(i+1, i+3) += Mcos_sin_d1(i+1, i+3);
                L_matrix(i, m-2) += Mcos_sin_d1(i, m-2);
            }
            else if(i==m-1)
            {
                L_matrix(i, m-3) += Mcos_sin_d1(i, m-3);
                L_matrix(i-1, m-4) += Mcos_sin_d1(i-1, m-3);
            }
            else if((i > 1) && (i < m-1))
            {
                L_matrix(i, i-2) += Mcos_sin_d1(i, i-2);
                L_matrix(i, i+2) += Mcos_sin_d1(i, i+2);

            }
        });
        

    }

    // Compute Laplacian matrix in Fourier space
    // and split into even and odd parts
    void even_odd_laplacian_matrix(view_2d<Real> Lo_matrix, view_2d<Real> Le_matrix, Real Kappa)
    {
        int nrows=Lo_matrix.extent(0);
        int ldims = 2*nrows;    // dimensions of full Laplacian matrix L_matrix
        view_2d<Real> L_matrix("temporaryL", ldims, ldims);

        // compute the Lmatrix using its known structure
        laplacian_matrix(L_matrix, Kappa);

         Kokkos::parallel_for(nrows, [=](Int i){
            Int v = nrows % 2;
            if(i == 0)
            {
                //odd part 
                Int j = 1 - v;
                Lo_matrix(i,i) = L_matrix(j,j);
                Lo_matrix(i, 1) = L_matrix(j, 3 - v);
                Lo_matrix(i, nrows-1) = L_matrix(j, 1 - v + 2*(nrows -1));

                //even part
                Int k = v;
                Le_matrix(i,i) = L_matrix(k,k);
                Le_matrix(i, 1) = L_matrix(k, v + 2);
                Le_matrix(i, nrows-1) = L_matrix(k, v + 2*(nrows -1));
            }
            else if(i == nrows -1)
            {
                // odd part
                Int j = 1 - v + 2*(nrows - 1);
                Lo_matrix(i,i) = L_matrix(j,j);
                Lo_matrix(i, i-1) = L_matrix(j, 1 - v + 2*(nrows - 2));
                Lo_matrix(i, 0) = L_matrix(j, 1-v);

                // even part
                Int k =  v + 2*(nrows - 1);
                Le_matrix(i,i) = L_matrix(k,k);
                Le_matrix(i, i-1) = L_matrix(k,  v + 2*(nrows - 2));
                Le_matrix(i, 0) = L_matrix(k, v);

            }
            else
            {
                // odd part
                Int j = 1 - v + 2*i;
                Lo_matrix(i,i) = L_matrix(j,j);
                Lo_matrix(i, i-1) = L_matrix(j, 1 - v + 2*(i-1));
                Lo_matrix(i, i+1) = L_matrix(j, 1 - v + 2*(i+1));

                // even part
                Int k = v + 2*i;
                Le_matrix(i,i) = L_matrix(k,k);
                Le_matrix(i, i-1) = L_matrix(k, v + 2*(i-1));
                Le_matrix(i, i+1) = L_matrix(k,  v + 2*(i+1));



            }
           
         });


        
    }

    
    


}