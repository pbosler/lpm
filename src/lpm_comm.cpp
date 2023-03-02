#include "lpm_comm.hpp"

#include "lpm_assert.hpp"

namespace Lpm {

Comm::Comm() {
  check_mpi_init();
  reset_mpi_comm(MPI_COMM_SELF);
}

Comm::Comm(MPI_Comm mpi_comm) {
  check_mpi_init();
  reset_mpi_comm(mpi_comm);
}

void Comm::reset_mpi_comm(MPI_Comm new_mpi_comm) {
  m_mpi_comm = new_mpi_comm;
  MPI_Comm_size(m_mpi_comm, &m_size);
  MPI_Comm_rank(m_mpi_comm, &m_rank);
}

void Comm::check_mpi_init() const {
  int flag;
  MPI_Initialized(&flag);
  LPM_ASSERT(flag != 0);
}

}  // namespace Lpm
