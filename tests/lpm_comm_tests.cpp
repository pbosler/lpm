#include <mpi.h>

#include <catch2/catch_test_macros.hpp>

#include "LpmConfig.h"
#include "lpm_comm.hpp"

using namespace Lpm;

TEST_CASE("default_comm_tests", "[mpi]") {
  Comm mcomm;

  REQUIRE(mcomm.rank() == 0);
  REQUIRE(mcomm.size() == 1);
  REQUIRE(mcomm.i_am_root());
}

TEST_CASE("mpi_comm_tests", "[mpi]") {
  Comm mcomm(MPI_COMM_WORLD);

  int mrank;
  int msize;

  MPI_Comm_rank(MPI_COMM_WORLD, &mrank);
  REQUIRE(mrank == mcomm.rank());
  MPI_Comm_size(MPI_COMM_WORLD, &msize);
  REQUIRE(msize == mcomm.size());

  if (mrank == 0) {
    REQUIRE(mcomm.i_am_root());
  } else {
    REQUIRE_FALSE(mcomm.i_am_root());
  }
}
