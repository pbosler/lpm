#include "dfs_doubling.hpp"
#include "dfs_laplacian_new.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <fftw3.h>
#include "dfs_rhs_new.hpp"
#include "dfs_solve_new.hpp"

using namespace SpherePoisson;

struct Input {
  int nrows;
  int ncols;
  GridType grid_type;

  Input(int argc, char* argv[]);

  std::string info_string() const;
};

int main(int argc, char* argv[]) {

  Input input(argc, argv);
  Kokkos::initialize(argc, argv);
{
  const int nrows = input.nrows;
  const int ncols = input.ncols;
  const auto dnrows = 2*(input.nrows - (input.grid_type == GridType::Shifted ? 0 : 1));

  std::cout << "nrows = " << nrows << " ncols = " << ncols << " dnrows = " << dnrows << "\n";

  // TODO: m rows, n columns?
  view_2d<Real> u("u", nrows, ncols);
  view_2d<Real> u_tilde("u_tilde", dnrows, ncols);
  Kokkos::parallel_for("initialize_u",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nrows, ncols}), KOKKOS_LAMBDA (const int i, const int j) {
    u(i,j) = i+j;
  });

  dfs_doubling(u_tilde, u, input.grid_type);
  
// TODO: test to make sure the doubling is correct
std::vector<Real> utildtest(dnrows*ncols);
std::ifstream ifs("../../datafiles/test_reflector.bin", std::ios::binary | std::ios::in);
//std::vector<std::complex<float>> v(3);
ifs.read(reinterpret_cast<char*>(utildtest.data()), (dnrows*ncols)*sizeof(double));
ifs.close();


 double err = -99.9;
 view_2d<Real>::HostMirror h_u_tilde = Kokkos::create_mirror_view( u_tilde);
 Kokkos::deep_copy(h_u_tilde, u_tilde);

 for(int i=0; i<dnrows; i++)
 {
  for(int j=0; j<ncols; j++)
  {
    err = fmax(fabs(h_u_tilde(i,j) - utildtest[i + j*ncols]), err);
   
  }
  std::cout<<std::endl; 
  
 }
 
if(err == 0)
{
  std::cout<<"Error is good"<<std::endl;
  
}
else{
  std::cout<<"Error is large";
  exit( -1);
}


} // kokkos scope
Kokkos::finalize();


}

Input::Input(int argc, char* argv[]) {
  nrows = 13;
  ncols = 24;
  int grid_type_int = 1;
  for (int i = 1; i < argc; ++i) {
    const std::string& token = argv[i];
    if (token == "-nrows") {
      nrows = std::stoi(argv[++i]);
    }
    else if (token == "-ncols") {
      ncols = std::stoi(argv[++i]);
    }
    else if (token == "-g") {
      grid_type_int = std::stoi(argv[++i]);
      
    }
    else {
      std::cerr << "received unknown argument: " << token << "\n";
    }
  }
  grid_type = static_cast<GridType>(grid_type_int);

}
