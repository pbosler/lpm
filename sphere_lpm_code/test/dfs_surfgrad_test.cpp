#include <cstdio>
#include <sstream>
#include <fftw3.h>
#include "dfs_surface_grad.hpp"
#include "dfs_test_support.hpp"


using namespace SpherePoisson;
// test function
void test_u(view_1d<Real> lat, view_1d<Real> lon, view_2d<Real> u);

// exact surface gradients
void surf_grad(view_1d<Real> lat, view_1d<Real> lon, view_2d<Real> du_dx,
view_2d<Real> du_dy, view_2d<Real> du_dz);


//
Real test_surf_grad(int nrows, int ncols);

int main(int argc, char * argv[])
{
    Kokkos::initialize(argc, argv);
    {
        int nrows = 12;
        int ncols = 24;
        Real err;
       
        err =  test_surf_grad(nrows,  ncols);
        
        if(err < 5e-14)
        {
            std::cout<<"Surface gradient computed correctly \n";
        }
        else{
            std::cout<<"Surface gradient computed incorrectly \n";
            exit(-1);
        }


        
        
    }
    // kokkos scope
    Kokkos::finalize();
    return 0;
}

// test function
void test_u(view_1d<Real> lat, view_1d<Real> lon, view_2d<Real> u)
{
    Int nrows = lat.extent(0);
    Int ncols = lon.extent(0);

     Kokkos::parallel_for("in",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrows, ncols}), KOKKOS_LAMBDA (const int i, const int j) {
            u(i,j) = sin(lat(i)) * cos(lat(i)) * sin(lon(j));

      });

}

//surface derivatives
void surf_grad(view_1d<Real> lat, view_1d<Real> lon, view_2d<Real> du_dx,
view_2d<Real> du_dy, view_2d<Real> du_dz)
{
    Int nrows = lat.extent(0);
    Int ncols = lon.extent(0);

    Kokkos::parallel_for("in",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrows, ncols}), KOKKOS_LAMBDA (const int i, const int j) {
            du_dx(i,j) = -sin(2*lon(j)) * cos(lat(i)) * pow(sin(lat(i)),2);

            du_dy(i,j) = pow(cos(lon(j)),2)*sin(lat(i)) 
            + pow(sin(lon(j)),2)*cos(2*lat(i))*cos(lat(i));

            du_dz(i,j) = -sin(lon(j))*cos(2*lat(i))*sin(lat(i));
      });
}


// testng routine
Real test_surf_grad(int nrows, int ncols)
{
    Int dnrows = 2*(nrows - 1);
    view_1d<Complex> cn("cn", dnrows);
    GridType grid_type = static_cast<GridType>(1);
    view_2d<Real> u("matrix", nrows, ncols);
    view_2d<Complex> U("matrix", dnrows, ncols);
    view_2d<Complex> du_dtheta("matrix", dnrows, ncols);
    view_2d<Complex> du_dlambda("matrix", dnrows, ncols);
    view_2d<Complex> Du_dx("matrix", dnrows, ncols);
    view_2d<Complex> Du_dy("du_dz", dnrows, ncols);
    view_2d<Complex> Du_dz("du_dz", dnrows, ncols);
    view_2d<Real> du_dx("matrix", nrows, ncols);
    view_2d<Real> du_dy("matrix", nrows, ncols);
    view_2d<Real> du_dz("matrix", nrows, ncols);
    view_2d<Real> du_dxr("matrix", nrows, ncols);
    view_2d<Real> du_dyr("matrix", nrows, ncols);
    view_2d<Real> du_dzr("matrix", nrows, ncols);
    view_1d<Real> lat("lat", nrows);
    view_1d<Real> lon("lon", ncols);
    view_1d<Complex> d_theta("dth", dnrows);
    view_1d<Complex> d_lambda("dlb", dnrows);
    
   // Initializations
    coords(lat, lon);
    test_u(lat, lon,  u);
    interp_shifts(grid_type, cn);
    vals2CoeffsDbl(grid_type, cn, u, U);
    dfs_diff_operators(d_theta, d_lambda);

    surf_grad(lat, lon, du_dxr, du_dyr, du_dzr);
   
    // Estimate the surface gradients
    dfs_diff_lat_lon(U, d_theta, d_lambda, du_dtheta,  du_dlambda);
    dfs_grad_u(du_dtheta, du_dlambda, Du_dx, Du_dy, Du_dz);

    // transform back to values
    coeffs2valsDbl(grid_type, cn, Du_dx, du_dx);
    coeffs2valsDbl(grid_type, cn, Du_dy, du_dy);
    coeffs2valsDbl(grid_type, cn, Du_dz, du_dz);


   
    Real err = 0.0;
    // KOKKOS_LAMBDA macro includes capture-by-value specifier [=].
    Kokkos::parallel_reduce("Reduction", nrows, KOKKOS_LAMBDA(const int i, Real& update) {
        Real col_sum = 0;
        for(int j =0; j<ncols; j++){
            col_sum = fmax(col_sum, abs(du_dx(i,j) - du_dxr(i,j))); 
            col_sum = fmax(col_sum, abs(du_dy(i,j) - du_dyr(i,j))); 
            col_sum = fmax(col_sum, abs(du_dz(i,j) - du_dzr(i,j))); 
        }
        update = fmax(col_sum, update);
   }, Kokkos::Max<Real>(err));


    std::cout<<"Error = "<<err<<std::endl;
    return err;
}