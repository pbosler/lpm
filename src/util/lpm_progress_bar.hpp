#ifndef LPM_PROGRESS_BAR_HPP
#define LPM_PROGRESS_BAR_HPP

#include <iostream>

#include "LpmConfig.h"

namespace Lpm {

class ProgressBar {
  std::string name_;
  Int niter_;
  Real freq_;
  Int it_;
  Real next_;
  std::ostream& os_;

 public:
  ProgressBar(const std::string& name, const Int niterations,
              const Real write_freq = 10.0, std::ostream& os = std::cout);

  void update();
};

}  // namespace Lpm

#endif
