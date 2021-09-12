#include "util/lpm_timer.hpp"
#include <sstream>

namespace Lpm {

std::string Timer::info_string() const {
  std::ostringstream ss;
  ss << "timer(" << _name << ") : elapsed time = " << _elap << " seconds\n";
  return ss.str();
}

}
