#ifndef LPM_TIMER_HPP
#define LPM_TIMER_HPP

#include "LpmConfig.h"

#include <sys/time.h>
#include <sys/resource.h>

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
static double get_memusage () {
  static const double scale = 1.0 / (1 << 10); // Memory in MB.
  rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  return ru.ru_maxrss*scale;
}
static void print_times (const std::string& name, const double* const parts,
                         const int nparts) {
  double total = 0; for (int i = 0; i < nparts; ++i) total += parts[i];
  printf("%20s %1.3e s %7.1f MB", name.c_str(), total, get_memusage());
  for (int i = 0; i < nparts; ++i) printf(" %1.3e s", parts[i]);
  printf("\n");
}
static void print_times (const std::string& name, const double total) {
   printf("%20s %1.3e s %5.1f MB\n", name.c_str(), total, get_memusage());

} // namespace Lpm

#endif
