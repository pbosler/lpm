#ifndef DFS_LHS_POISSON_HPP
#define DFS_LHS_POISSON_HPP

#include "dfs_config.hpp"

namespace SpherePoisson{
    class LHS_Poisson{
        private:
            Int nrow;
            view_2d<Real> L_matrix;
            view_2d<Real> sin_d2;
            view_2d<Real> Mcos_sin_d1;

            // private setter method
            void set_sind2();
            void set_Mcossind1();
            void set_Lmatrix();
            void split_Lmatrix();

        public:
            view_2d<Real> Lo_matrix;
            view_2d<Real> Le_matrix;
            Real Kappa;

            // constructor
            LHS_Poisson(Int nrow_, Real Kappa_);
            
            // default destructor
            ~LHS_Poisson(){}


    };

}
#endif


