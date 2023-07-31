#include "dfs_vort2velocity.hpp"


using namespace SpherePoisson;

/*
    assume vorticity is 2sin(colat)*sin(lon)
    this gives stream function -sin(colat)*sin(lon)
*/
void grid_vort(view_1d<Real> lat, view_1d<Real> lon, view_2d<Real> vort);

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
//
void grid_vort(view_1d<Real> lat, view_1d<Real> lon, view_2d<Real> vort)
{
    Int nrows = lat.extent(0);
    Int ncols = lon.extent(0);

    Kokkos::parallel_for("initialize_u",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrows, ncols}), KOKKOS_LAMBDA (const int i, const int j) {
        vort(i,j) = 2 * sin(lat(i)) * sin(lon(j));
  });

}
//
 void my_velocity(view_1d<Real> lat, view_1d<Real> lon, view_1d<Real> u, view_1d<Real> v, view_1d<Real> w)
 {
        Int nrows = lat.extent(0);
    
        Kokkos::parallel_for(nrows,  KOKKOS_LAMBDA (const int i) {
           u(i) = cos(lat(i));
           v(i) = 0.0;
           w(i) = -cos(lon(i)) * sin(lat(i));
        });
}
//
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

//
Real test_interpolant(Int nrows, int ncols)
{
    Int dim1 = 1000;
    Int dim2 = 2000;
    const GridType grid_type = static_cast<GridType>(1);
    view_1d<Real> lat("lat", nrows);
    view_1d<Real> lon("lon", ncols);
    view_2d<Real> vort("vort", nrows, ncols);  // the vorticity

    // particle positions: cartesian
    view_r3pts<Real> X("X[x,u,z]",dim1*dim2);

    // particle position in spherical coordinates
    view_1d<Real> th("x", dim1*dim2);
    view_1d<Real> lb("y", dim1*dim2);
    

    view_1d<Real> uxy("interpolant", dim1*dim2);
    view_1d<Real> u_true("true f(x,y)", dim1*dim2);
    view_1d<Real> vxy("interpolant", dim1*dim2);
    view_1d<Real> v_true("true f(x,y)", dim1*dim2);
    view_1d<Real> wxy("interpolant", dim1*dim2);
    view_1d<Real> w_true("true f(x,y)", dim1*dim2);
    
    // compute the coordinate on the colatitute-longitude grid
    coords(lat, lon);

    // evaluate the vorticity on the computed colatitude-longitude
    grid_vort(lat, lon,  vort);

    
    // Evaluation points
    Int M = dim1 * dim2;

     Kokkos::parallel_for(M, KOKKOS_LAMBDA (const int i) {
    
            Real xx = ((Real)rand() / RAND_MAX);
            Real yy = ((Real)rand() / RAND_MAX);
            Real zz = ((Real)rand() / RAND_MAX);
            
            // Ensure the points are located on the unit sphere 
            if(xx*xx + yy*yy +zz*zz)
            {
                X(i, 0) = xx;
                X(i, 1) = yy;
                X(i, 2) = zz;

                // convert the cartesian coordinates to spherical coordinates
                th(i) = atan2(sqrt(xx*xx + yy*yy), zz);
                lb(i) = atan2(yy, xx);

            }

        });

        /*
            Compute the velocity for each particle
            given the vorticity on the sphere
            and cartesian coordinates of each particle on the sphere
        */
       view_r3pts<Real> U_X("U_X", M);
       dfs_vort_2_velocity(X, vort, U_X);
     
       // get true value of at the random points
        my_velocity(th, lb, u_true, v_true, w_true);


        Real my_err =  error_fun(U_X, u_true, v_true, w_true);
      

    return my_err;
}

