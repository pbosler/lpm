#ifndef DFS_THREADS_HPP
#define DFS_THREADS_HPP

#include <fftw3.h>

#include <cstdlib>
#include <mutex>

#include "dfs_config.hpp"

namespace SpherePoisson {

/** Thread counts for the third-party transforms used by the DFS solver.

  The FFTs and NUFFTs are small (grids of a few thousand points) and are called
  several times per time step.  Profiling showed finufft's execute step running
  about 6x slower with 2 or more threads than with 1, so both libraries default
  to a single thread.  Set LPM_NUFFT_THREADS / LPM_FFTW_THREADS to override
  (0 means all of the OpenMP/Kokkos threads).
*/
inline int env_thread_count(const char* name, const int default_value) {
  const char* value = std::getenv(name);
  return (value ? std::atoi(value) : default_value);
}

/// number of threads for finufft (finufft_opts::nthreads; 0 = all)
inline int nufft_threads() {
  static const int n = env_thread_count("LPM_NUFFT_THREADS", 1);
  return n;
}

/// number of threads for FFTW plans; initializes FFTW's thread support once.
inline int fftw_threads() {
  static const int n = [] {
    int requested = env_thread_count("LPM_FFTW_THREADS", 1);
    if (requested <= 0) requested = Lpm::DevExe().concurrency();
    fftw_init_threads();
    return requested;
  }();
  return n;
}

}  // namespace SpherePoisson

#endif
