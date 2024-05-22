#ifndef DFS_SOLVE_NEW_HPP
#define DFS_SOLVE_NEW_HPP
#include <fftw3.h>

#include "KokkosBlas.hpp"
#include "dfs_config.hpp"


namespace SpherePoisson {
    // Thomas algorithm: the tridiagonal solver
     // In our application we expect real RHS and
     // Complex left handside
    template<typename T>
    void tridiag_solver(view_2d<T> A, view_2d<Complex> d, Int k);

    // the triadiagonal solver when we have a stuck of two-dimensional arrays
    template<typename T>
    void tridiag_solver_3d(view_3d<T> A, view_2d<Complex> d, Int k);

    // Fixes the singularity in rhse when K=0 (Poisson) for even zero mode
    // It also rearange the matrix Le so that the dense row is
    // at the bottom to avoid high fill during Gaussian elimination
    void rearrange(view_2d<Complex> Le, view_1d<Complex> fe, view_1d<Int> ie);

    // Solver for the zero even mode when the problem is Poisson
    // needs a special tridiag_solver
    // fe is the right handside and will be replaced by the solution
    void special_solver(view_2d<Complex> Le, view_1d<Complex> fe, view_1d<Int> ie);


    // Solver
    void solver(view_2d<Real> Le, view_2d<Real> Lo, view_2d<Complex> rhse,  view_2d<Complex> rhso, view_2d<Complex> UU,
    view_1d<Int> io, view_1d<Int> ie, Real Kappa);




}
#endif
