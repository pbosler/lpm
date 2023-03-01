#include "lpm_test_utils.hpp"

namespace Lpm {

int get_test_device(const int mpi_rank) {
  // Set to -1 by default, which leaves kokkos in full control
  int dev_id = -1;

  // TODO: Fill in with ekat stuff when needed

  return dev_id;
}

}  // namespace Lpm
