#include "dfs_interpolation.hpp"

using namespace SpherePoisson;

void my_velocity(view_1d<Real> lat, view_1d<Real> lon, view_1d<Real> u, view_1d<Real> v, view_1d<Real> w);

Real  error_fun(view_r3pts<Real> U_X, view_1d<Real> u, view_1d<Real> v, view_1d<Real> w);

Real test_interpolant(Int nrows, int ncols);

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        // sampling f at uniform points and compute 
        // its Fourier coefficients using DFS method
        Int nrows = 17;
        Int ncols = 32;
        
       Real err = test_interpolant(nrows, ncols);



    if(err<5e-14)
    {
        std::cout<<"Interpolation done correctly and error = "<<err<<std::endl;
    }
    else
    {
        std::cout<<"Interpolation done incorrectly and error = "<<err<<std::endl;
        exit(-1);
    }
    // kokkos scope
    }
    Kokkos::finalize();
    return 0;

}

 void my_velocity(view_1d<Real> lat, view_1d<Real> lon, view_1d<Real> u, view_1d<Real> v, view_1d<Real> w)
 {
        Int nrows = lat.extent(0);
    
        Kokkos::parallel_for(nrows,  KOKKOS_LAMBDA (const int i) {
           u(i) = -cos(lat(i)) * (3*cos(lon(i))*cos(3*lon(i)) + sin(lon(i))*sin(3*lon(i)));
           v(i) = -cos(lat(i))*(3*sin(lon(i))*cos(3*lon(i)) - cos(lon(i))*sin(3*lon(i)));
           w(i) = 3*cos(3*lon(i)) * sin(lat(i));
        });
}

Real  error_fun(view_r3pts<Real> U_X, view_1d<Real> u, view_1d<Real> v, view_1d<Real> w)
{
    Int N = u.extent(0);
    Real err = 0.0;

    Kokkos::parallel_reduce(N, [=](Int j, Real& maxval){
         
        Real val = fmax(abs(U_X(j,0) - u(j)),abs(U_X(j,1) - v(j)));
        Real val2 = fmax(abs(U_X(j,2) - w(j)), val);
        maxval = fmax(val2, maxval);
           
       
     }, Kokkos::Max<Real>(err));

     return err;
}


Real test_interpolant(Int nrows, int ncols)
{
    Int dnrows = 2*(nrows - 1);
    Int dim1 = 1000;
    Int dim2 = 2000;
    const GridType grid_type = static_cast<GridType>(1);
    view_1d<Real> lat("lat", nrows);
    view_1d<Real> lon("lon", ncols);
    view_2d<Real> u("f_ij", nrows, ncols);
    view_2d<Complex> U("coeffs of f", dnrows, ncols);
    view_2d<Real> v("f_ij", nrows, ncols);
    view_2d<Complex> V("coeffs of f", dnrows, ncols);
    view_2d<Real> w("f_ij", nrows, ncols);
    view_2d<Complex> W("coeffs of f", dnrows, ncols);
    view_1d<Complex> cn("cn", dnrows);
    view_1d<Real> x("x", dim1*dim2);
    view_1d<Real> y("y", dim1*dim2);
    view_1d<Real> uxy("interpolant", dim1*dim2);
    view_1d<Real> u_true("true f(x,y)", dim1*dim2);
    view_1d<Real> vxy("interpolant", dim1*dim2);
    view_1d<Real> v_true("true f(x,y)", dim1*dim2);
    view_1d<Real> wxy("interpolant", dim1*dim2);
    view_1d<Real> w_true("true f(x,y)", dim1*dim2);

    interp_shifts(grid_type, cn);
    coords(lat, lon);
    true_velocity(lat, lon,  u, v, w);

    // Computing Fourier  coefficients of the velocities
    vals2CoeffsDbl(grid_type, cn, u, U);
    vals2CoeffsDbl(grid_type, cn, v, V);
    vals2CoeffsDbl(grid_type, cn, w, W);

    // Evaluation points
    Int M = dim1 * dim2;

     Kokkos::parallel_for(M, KOKKOS_LAMBDA (const int i) {
    
            x(i) = M_PI * ((Real)rand() / RAND_MAX);;
            y(i) = 2*M_PI *((Real)rand() / RAND_MAX);
            
        });

        // Evaluate the interpolant
       view_r3pts<Real> U_X("U_X", M);
       dfs_interp(U, V, W, x, y, U_X);
     
        // get true value of at the random points
        my_velocity(x, y, u_true, v_true, w_true);


      Real my_err =  error_fun(U_X, u_true, v_true, w_true);
      for(int i=0; i<10; i++)
      {
        std::cout<<x(i)<<y(i)<<U_X(i,2)<<std::endl;
      }


    return my_err;
}

