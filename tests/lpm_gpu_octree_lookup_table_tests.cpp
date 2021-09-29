#include <bitset>
#include <iostream>
#include "LpmConfig.h"
#include "lpm_coords.hpp"
#include "lpm_logger.hpp"
#include "lpm_comm.hpp"
#include "lpm_constants.hpp"
#include "tree/lpm_gpu_octree_functions.hpp"
#include "tree/lpm_gpu_octree_lookup_tables.hpp"
#include "util/lpm_math.hpp"
#include "util/lpm_floating_point.hpp"
#include "catch.hpp"

using namespace Lpm;
using namespace Lpm::tree;

TEST_CASE("gpu_octree_lookup_tables", "[tree]") {

  Comm comm;

  Logger<> logger("gpu_lookup_tables", Log::level::info, comm);

  Kokkos::View<ParentLUT> parent_table("parent_table");
  Kokkos::View<ChildLUT> child_table("child_table");

  SECTION("parent/child tables") {

    // identity tests
    Kokkos::View<Int[8]> parent_identity("parent_identity");
    Kokkos::View<Int[8]> child_identity("child_identity");
    Kokkos::parallel_for(8, KOKKOS_LAMBDA (const int i) {
      const auto j = 13;
      // i = child idx
      // j = 13 (self) = neighbor index
      // assert that the parent of myself (j=13) is my parent
      parent_identity(i) = table_val(i, j, parent_table);
      // assert that myself (j=13) is the ith child of my parent
      child_identity(i) = table_val(i, j, child_table);
    });
    auto h_p_i = Kokkos::create_mirror_view(parent_identity);
    auto h_c_i = Kokkos::create_mirror_view(child_identity);
    Kokkos::deep_copy(h_p_i, parent_identity);
    Kokkos::deep_copy(h_c_i, child_identity);

    for (auto i=0; i<8; ++i) {
      REQUIRE(h_p_i(i) == 13);
      REQUIRE(h_c_i(i) == i);
    }
  }

  SECTION("connectivity tables") {

  }

}
