#ifndef LPM_TIMER_HPP
#define LPM_TIMER_HPP

#include "LpmConfig.h"
#include <mpi.h>
#include "Kokkos_Core.hpp"
#include <sys/time.h>
#include <sys/resource.h>
#include <string>

namespace Lpm {

/// Timers
static timeval tic () {
  timeval t;
  gettimeofday(&t, 0);
  return t;
}
static double calc_et (const timeval& t1, const timeval& t2) {
  static constexpr double us = 1.0e6;
  return (t2.tv_sec * us + t2.tv_usec - t1.tv_sec * us - t1.tv_usec) / us;
}
static double toc (const timeval& t1) {
  Kokkos::fence();
  timeval t;
  gettimeofday(&t, 0);
  return calc_et(t1, t);
}

struct Timer {
    Timer(const std::string& name="") : _name(name), _start(0), _end(0) {}

    inline void start() {_start = MPI_Wtime();}

    inline void stop() {_end = MPI_Wtime(); _elap = _end - _start;}

    inline Real elapsed() const {return _elap;}

    std::string info_string() const;

  private:
    Real _start;
    Real _end;
    Real _elap;
    std::string _name;

};

} // namespace Lpm

#endif
