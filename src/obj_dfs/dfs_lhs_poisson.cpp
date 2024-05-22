#include "dfs_lhs_poisson.hpp"

namespace SpherePoisson
{
    LHS_Poisson::LHS_Poisson(Int nrow_, Real Kappa_)
    {
        nrow = nrow_;
        Kappa = Kappa_;
        Le_matrix(nrow, nrow);
        Lo_matrix(nrow, nrow);
        L_matrix(2*nrow, 2*nrow);
        sin_d2(2*nrow, 2*nrow);
        Mcos_sin_d1(2*nrow, 2*nrow);

    }

    // implementing the other methods
    void LHS_Poisson::set_sind2()
    {
        Int m = 2*nrow;
         // For Poisson K is zero
        if(Kappa == 0)
        {
            
            Kokkos::parallel_for(m, [=](Int i){
    
            Real a = -0.125;
            Real b = -0.25;
            Real c = 0.5;
            Real dd = double(-m/2.0);
            
            if(i==0)
            {
                sin_d2(i,i) = -c * pow(dd,2);
                sin_d2(i, i+2) = -b * pow((dd + (i+2)),2);
                sin_d2(i, m-2) = -b * pow((dd + (m-2)),2);
            }
            else if(i==1)
            {
                sin_d2(i,i) = -c * pow((dd + i),2);
                sin_d2(i, i+2) = -b * pow((dd + i + 2),2);
            }
            else if(i==m-2)
            {
                sin_d2(i,i) = -c * pow((dd + i),2);
                sin_d2(i, i-2) = -b * pow((dd + i - 2),2);
                sin_d2(i, 0) = -a * pow(dd,2);

            }
            else if(i==m-1)
            {
                sin_d2(i, i-2) = -b * pow((dd + i - 2),2);
                sin_d2(i,i) = -c * pow((dd + i),2);
            }
            else
            {
                Real d =  i == 2 ? 1.0 : 2.0;
                sin_d2(i,i) = -c * pow(dd + i,2);
                sin_d2(i,i-2) = -d * a * pow((dd + i - 2),2);
                sin_d2(i, i+2) = -b * pow((dd + i + 2),2);

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
                sin_d2(i,i) = c * (-pow(dd,2) + k2);
                sin_d2(i, i+2) = b * (-pow(dd + (i+2),2) + k2);
                sin_d2(i, m-2) = b * (-pow(dd + (m-2),2) + k2);
            }
            else if(i==1)
            {
                sin_d2(i,i) = c * (-pow(dd + i,2) + k2);
                sin_d2(i, i+2) = b * (-pow(dd + i + 2,2) + k2);
            }
            else if(i==m-2)
            {
                sin_d2(i,i) = c * (-pow(dd + i,2) + k2);
                sin_d2(i, i-2) = b * (-pow(dd + i - 2,2) + k2);
                sin_d2(i, 0) = a * (-pow(dd,2) + k2);

            }
            else if(i==m-1)
            {
                sin_d2(i, i-2) = b * (-pow(dd + i - 2,2) + k2);
                sin_d2(i,i) = c * (-pow(dd + i,2) + k2);
            }
            else
            {
                Real d =  i == 2 ? 1.0 : 2.0;
                sin_d2(i,i) = c * (-pow(dd + i,2) + k2);
                sin_d2(i,i-2) = d * a * (-pow(dd + i - 2,2) + k2);
                sin_d2(i, i+2) = b * (-pow(dd + i + 2,2) + k2);
            }
            });
        }
    }

    void LHS_Poisson::set_Mcossind1()
    {
        Int m = 2*nrow;
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
        

        });
    }

    void LHS_Poisson::set_Lmatrix()
    {
        Int m = 2*nrow;
        Kokkos::parallel_for(m, [=](Int i){
            if(i == 0)
            {
                L_matrix(i, i+2) = sin_d2(i, i+2) + Mcos_sin_d1(i, i+2);
                L_matrix(i+1, i+3) = sin_d2(i, i+3) + Mcos_sin_d1(i+1, i+3);
                L_matrix(i, m-2) = sin_d2(i, m-2) + Mcos_sin_d1(i, m-2);
            }
            else if(i==m-1)
            {
                L_matrix(i, m-3) = sin_d2(i, m-3) + Mcos_sin_d1(i, m-3);
                L_matrix(i-1, m-4) = sin_d2(i, m-4) + Mcos_sin_d1(i-1, m-4); //m-3?
            }
            else if((i > 1) && (i < m-1))
            {
                L_matrix(i, i-2) = sin_d2(i, i-2) + Mcos_sin_d1(i, i-2);
                L_matrix(i, i+2) = sin_d2(i, i+2) + Mcos_sin_d1(i, i+2);

            }
        });
        
    }

   void LHS_Poisson::split_Lmatrix()
   {
        Kokkos::parallel_for(nrow, [=](Int i){
            Int v = nrow % 2;
            if(i == 0)
            {
                //odd part 
                Int j = 1 - v;
                Lo_matrix(i,i) = L_matrix(j,j);
                Lo_matrix(i, 1) = L_matrix(j, 3 - v);
                Lo_matrix(i, nrow-1) = L_matrix(j, 1 - v + 2*(nrow -1));

                //even part
                Int k = v;
                Le_matrix(i,i) = L_matrix(k,k);
                Le_matrix(i, 1) = L_matrix(k, v + 2);
                Le_matrix(i, nrow-1) = L_matrix(k, v + 2*(nrow -1));
            }
            else if(i == nrow -1)
            {
                // odd part
                Int j = 1 - v + 2*(nrow - 1);
                Lo_matrix(i,i) = L_matrix(j,j);
                Lo_matrix(i, i-1) = L_matrix(j, 1 - v + 2*(nrow - 2));
                Lo_matrix(i, 0) = L_matrix(j, 1-v);

                // even part
                Int k =  v + 2*(nrow - 1);
                Le_matrix(i,i) = L_matrix(k,k);
                Le_matrix(i, i-1) = L_matrix(k,  v + 2*(nrow - 2));
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
} // namespace Sp

