#include "LpmConfig.h"
#include "LpmDefs.hpp"

#include "Kokkos_Core.hpp"
#include "KokkosBlas.hpp"

#include <iostream>
#include <exception>

using namespace Lpm;

int main(int argc, char* argv[]) {
ko::initialize(argc, argv);
{
  const Int nn = 100;
  scalar_view_type ones("ones", nn);
  scalar_view_type twos("twos", nn);

  KokkosBlas::fill(ones, 1.0);
  KokkosBlas::fill(twos, 2.0);

  KokkosBlas::axpy(2.0, ones, twos);

  int nerr = 0;
  auto host_twos = ko::create_mirror_view(twos);
  ko::deep_copy(host_twos, twos);
  for (int i=0; i<nn; ++i) {
    if ( std::abs(host_twos(i) - 4) > ZERO_TOL) {
      ++nerr;
      std::cout << "host_twos(i) = " << host_twos(i) << "\n";
    }
  }

  if (nerr == 0) {
    std::cout << "2 + 2 = 4; Tests pass.\n";
  }
  else {
    throw std::logic_error("2+2 != 4");
  }
}
ko::finalize();
return 0;
}
