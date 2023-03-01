#ifndef LPM_COMM_HPP
#define LPM_COMM_HPP

#include <mpi.h>

#include "LpmConfig.h"

namespace Lpm {

class Comm {
 public:
  Comm();

  explicit Comm(MPI_Comm mpi_comm);

  void reset_mpi_comm(MPI_Comm new_mpi_comm);

  bool i_am_root() const { return m_rank == 0; }

  int rank() const { return m_rank; }

  int size() const { return m_size; }

 private:
  void check_mpi_init() const;

  int m_rank;
  int m_size;
  MPI_Comm m_mpi_comm;
};

}  // namespace Lpm

#endif
