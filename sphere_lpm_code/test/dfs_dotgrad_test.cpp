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

// exact double dot of the gradient
void test_ddot(view_1d<Real> lat, view_1d<Real> lon, view_2d<Real> dgraddot);


//
Real test_Uddot(int nrows, int ncols);

int main(int argc, char * argv[])
{
    Kokkos::initialize(argc, argv);
    {
        int nrows = 13;
        int ncols = 24;
        Real err;
       
        err =  test_Uddot(nrows,  ncols);
        
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

// ddot
void test_ddot(view_1d<Real> lat, view_1d<Real> lon, view_2d<Real> dgraddot)
{
    Int nrows = lat.extent(0);
    Int ncols = lon.extent(0);
    
    Kokkos::parallel_for("doubledot",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrows, ncols}), KOKKOS_LAMBDA (const int i, const int j) {

        Real du_dx = -sin(2*lon(j)) * cos(lat(i)) * pow(sin(lat(i)),2);
        Real du_dy =  pow(cos(lon(j)),2)*sin(lat(i)) 
                    + pow(sin(lon(j)),2)*cos(2*lat(i))*cos(lat(i));
        Real du_dz = -sin(lon(j))*cos(2*lat(i))*sin(lat(i));
        Real tmp1 = du_dx*du_dx +  du_dy*du_dy +  du_dz*du_dz;
        Real tmp2 = 2.0 * (du_dy*du_dx + du_dz*du_dx + du_dz*du_dy);
        dgraddot(i,j) = tmp1 + tmp2;
          
     });
}


// testng routine
Real test_Uddot(int nrows, int ncols)
{
    Int dnrows = 2*(nrows - 1);
    view_1d<Complex> cn("cn", dnrows);
    GridType grid_type = static_cast<GridType>(1);
    view_2d<Real> u("matrix", nrows, ncols);
    view_2d<Complex> U("matrix", dnrows, ncols);
    view_1d<Real> lat("lat", nrows);
    view_1d<Real> lon("lon", ncols);
    view_2d<Complex> gradU_ddot("cdot", dnrows, ncols);
    view_2d<Real> uddot("ddot", nrows, ncols);
     view_2d<Real> uddot_true("ddot", nrows, ncols);
   
    
   // Initializations
    coords(lat, lon);
    test_u(lat, lon,  u);
    interp_shifts(grid_type, cn);
    vals2CoeffsDbl(grid_type, cn, u, U);
   
    // compute the double dot product
    double_dot_gradU(U, U,  U,  gradU_ddot);

    // transform back to values
    coeffs2valsDbl(grid_type, cn, gradU_ddot, uddot);

    // true double dot product
    test_ddot(lat, lon, uddot_true);
    
   
    Real err = 0.0;
    // KOKKOS_LAMBDA macro includes capture-by-value specifier [=].
    Kokkos::parallel_reduce("Reduction", nrows, KOKKOS_LAMBDA(const int i, Real& update) {
        Real col_sum = 0;
        for(int j =0; j<ncols; j++){ 
            col_sum = fmax(col_sum, abs(uddot(i,j) - uddot_true(i,j))); 
        }
        update = fmax(col_sum, update);
   }, Kokkos::Max<Real>(err));

   for(int i=0; i<nrows; i++)
   {
    for(int j=0; j<ncols; j++)
    {
        std::cout<<uddot(i,j)<<"    "<<uddot_true(i,j)<<std::endl;
    }
   }


    std::cout<<"Error = "<<err<<std::endl;
    return err;
}