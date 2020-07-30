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
      std::cout << "host_twos("<< i << ") = " << host_twos(i) << "\n";
    }
  }

  if (nerr == 0) {
    std::cout << "2 + 2 = 4; axpy test passes.\n";
  }
  else {
    std::cout << "KokkosBlas::axpy error: 2+2 != 4\n";
  }

  typedef ko::View<Real*[3]> crd_view_type;
  crd_view_type x("x",  nn);
  crd_view_type x1("x1",nn);
  crd_view_type x2("x2",nn);

  const int ne1 = nerr;

  KokkosBlas::fill(x, 3.0);
  KokkosBlas::fill(x1, 2.0);
  KokkosBlas::fill(x2, 100.0);

  KokkosBlas::update(1.0, x, 0.5, x1, 0.0, x2);
  auto host_fours = ko::create_mirror_view(x2);
  ko::deep_copy(host_fours, x2);
  for (Int i=0; i<nn; ++i) {
    for (Int j=0; j<3; ++j) {
      if (std::abs(host_fours(i,j) - 4) > ZERO_TOL) {
        ++nerr;
        std::cout << "host_fours(" << i << "," << j <<") = " << host_fours(i,j) << "\n";
      }
    }
  }
  if (nerr == ne1) {
    std::cout << "3 + 1 = 4; update test passes.\n";
  }
  else {
    std::cout << "KokkosBlas::update error 3+1 != 4\n";
  }

  const int ne2 = nerr;
  ko::View<Real*> sixes("sixes",nn);
  KokkosBlas::scal(sixes, 2, ko::subview(x, ko::ALL(), 0));
  auto host_sixes = ko::create_mirror_view(sixes);
  ko::deep_copy(host_sixes, sixes);
  for (Int i=0; i<nn; ++i) {
    if (std::abs(host_sixes(i) -6.0) > ZERO_TOL) {
      ++nerr;
      std::cout << "host_sixes(" << i << ") = " << host_sixes(i) << "\n";
    }
  }
  if (nerr == ne2) {
    std::cout << "2*3 = 6; scal test passes.\n";
  }
  else {
    std::cout << "KokkosBlas::scal error 2*3 != 6\n";
  }

  if (nerr > 0) {
    std::cout << "errors found: " << nerr << "\n";
    throw std::logic_error("fail: at least one test did not pass.");
  }
}
ko::finalize();
return 0;
}
