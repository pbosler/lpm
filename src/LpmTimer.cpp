#include "LpmTimer.hpp"
#include <sstream>

namespace Lpm {

std::string Timer::infoString() const {
  std::ostringstream ss;
  ss << "timer(" << _name << ") : elapsed time = " << _elap << " seconds\n";
  return ss.str();
}

}
