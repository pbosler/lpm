#ifndef DFS_VELOCITY_HPP
#define DFS_VELOCITY_HPP

#include "dfs_config.hpp"
#include<KokkosBlas1_axpby.hpp>



namespace SpherePoisson {
    // enumeration for trig function to chose between multiplying by sine and cosine
    enum trig{Sin, Cos};

    // Computes the velocity component w given in Fourier space give
    // Fourier coefficients of the  vorticity
    void  differentiate(view_2d<Complex> d_vort_theta, view_2d<Complex> d_vort_lambda);


    // Multiplying a function by cos(2theta) in Fourier space
    void multi_cos2theta(view_2d<Complex> mat, view_2d<Complex> cos2theta_mat);


    // Multiply by sin/cos(lambda) in Fourier
    // i.e. Multiply  A * sin/cos(lambda)
    //void multi_trig(view_2d<Complex> mat, view_2d<Complex> res, std::string str);
     void multi_trig(view_2d<Complex> mat, view_2d<Complex> res, Int trigsign);

    // Multiply by cos(theta) on the lefthandside
    // i.e. cos(theta) * A
    void multi_cos(view_2d<Complex> mat, view_2d<Complex> res);

    // Function for dividing by sin(theta) in
    // Fourier space
    void divide_sin(view_2d<Complex> mat, view_2d<Complex> res);

    // Given vorticity on the grid compute velocity (u, v, w)
    // on the grid
    void velocity_on_grid(view_2d<Complex> vort, view_2d<Complex> U,  view_2d<Complex> V, view_2d<Complex> W);


}
#endif
