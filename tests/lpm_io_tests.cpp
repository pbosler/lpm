#include "LpmConfig.h"
#include "util/lpm_math.hpp"
#include "util/lpm_floating_point.hpp"
#include "lpm_comm.hpp"
#include "lpm_constants.hpp"
#include "lpm_geometry.hpp"
#include "lpm_logger.hpp"
#include "util/lpm_numpy_io.hpp"
#include "util/lpm_matlab_io.hpp"
#include <catch2/catch_test_macros.hpp>
#include <typeinfo>
#include <fstream>
#include <string>

using namespace Lpm;

TEST_CASE("array/matrixoutput", "") {

  constexpr int M = 20;
  constexpr int N = 10;

  scalar_view_type ones("ones",N);
  Kokkos::deep_copy(ones, 1);
  auto h_ones = Kokkos::create_mirror_view(ones);
  Kokkos::deep_copy(h_ones, ones);

  Kokkos::View<Real**> twos("twos", M, N);
  Kokkos::deep_copy(twos, 2);
  auto h_twos = Kokkos::create_mirror_view(twos);
  Kokkos::deep_copy(h_twos, twos);

  std::vector<Real> threes(N, 3);

  SECTION("matlab") {
    std::ofstream ofile("lpm_matlab_test.m");

    write_vector_matlab(ofile, "ones", h_ones);
    write_array_matlab(ofile, "twos", h_twos);
    write_vector_matlab(ofile, "threes", threes);

    ofile.close();
  }

  SECTION("numpy") {
    std::ofstream ofile("lpm_numpy_test.py");
    numpy_import(ofile);
    write_vector_numpy(ofile, "ones", h_ones);
    write_array_numpy(ofile, "twos", h_twos);
    write_vector_numpy(ofile, "threes", threes);
    ofile.close();
  }
}

