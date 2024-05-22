#ifndef DFS_SOLVER_HPP
#define DFS_SOLVER_HPP

#include "dfs_lhs_poisson.hpp"
#include "dfs_rhs.hpp"

namespace SpherePoisson
{
    template<typename srcType, typename ViewType>
    class DFSSolver{
        private:
            void tridiag_solver(view_2d<Real> A, ViewType D, Int k );
            void tridiag_solver3d(view_3d<Real> A, ViewType D, Int k);
            void rearrange(view_2d<Complex> Le, view_1d<Complex> fe, view_1d<Int>);
            void special_solver(view_2d<Complex>Le, view_1d<Complex> fe, view_1d<Int> ie);
        
        public:
            void solver(LHS_Poisson& lhs_obj, DFS_RHS<srcType,ViewType> & rhs_obj, view_2d<Complex>UU);


    };
}


#endif