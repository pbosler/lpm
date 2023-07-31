#include "dfs_test_support.hpp"

namespace SpherePoisson {
    void coords(view_1d<Real> lat, view_1d<Real> lon)
    {
        Int nrows = lat.extent(0);
        Int ncols = lon.extent(0);

        Kokkos::parallel_for(nrows, [=](Int i){
            lat(i) = i*M_PI/(nrows-1);
        });

        Kokkos::parallel_for(ncols, [=](Int j){
            lon(j) = 2*j*M_PI/ncols;
        });
    }

    void my_vort(view_1d<Real> lat, view_1d<Real> lon, view_2d<Real> vort)
    {
        Int nrows = lat.extent(0);
        Int ncols = lon.extent(0);

         Kokkos::parallel_for("initialize",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrows, ncols}), KOKKOS_LAMBDA (const int i, const int j) {
            vort(i,j) = sin(lat(i))*sin(3*lon(j));
         });
    }

     void true_velocity(view_1d<Real> lat, view_1d<Real> lon, view_2d<Real> u, view_2d<Real> v, view_2d<Real> w)
     {
        Int nrows = lat.extent(0);
        Int ncols = lon.extent(0);

        Kokkos::parallel_for("initialize",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrows, ncols}), KOKKOS_LAMBDA (const int i, const int j) {
           u(i, j) = -cos(lat(i)) * (3*cos(lon(j))*cos(3*lon(j)) + sin(lon(j))*sin(3*lon(j)));
           v(i, j) = -cos(lat(i))*(3*sin(lon(j))*cos(3*lon(j)) - cos(lon(j))*sin(3*lon(j)));
           w(i, j) = 3*cos(3*lon(j)) * sin(lat(i));
        });
     }

     template<typename T>
    Real max_error(view_2d<T> u_true, view_2d<T> u)
    {
        Real err = 0.0;
        Int nrows = u_true.extent(0);
        Int ncols = u_true.extent(1);

         Kokkos::parallel_reduce(ncols, [=](Int j, Real& maxval){
         
            for(int i =0; i<nrows; i++)
            {
                Real val = abs(u_true(i,j) - u(i,j));
                maxval = fmax(val, maxval);
            }
       
     }, Kokkos::Max<Real>(err));


        return err;
    }


template Real max_error<Complex>(view_2d<Complex> exact, view_2d<Complex> estimate);
template Real max_error<Real>(view_2d<Real> exact, view_2d<Real> estimate);
template Real max_error<Int>(view_2d<Int> exact, view_2d<Int> estimate);


 // Test function for dfs interpolation
 void test_fun(view_1d<Real> lat, view_1d<Real> lon, view_2d<Real> f)
 {
    Int nrows = lat.extent(0);
    Int ncols = lon.extent(0);

    Kokkos::parallel_for("initialize",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrows, ncols}), KOKKOS_LAMBDA (const int i, const int j) {
            Real x = cos(lon(j))*sin(lat(i));
            Real y = sin(lon(j))*sin(lat(i));
            Real z = cos(lat(i));
           
            f(i,j) = x + y + z;
         });
    
 }



}