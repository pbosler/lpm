#include "dfs/lpm_dfs_polar_vortex_solver.hpp"
#include "lpm_logger.hpp"
#include "util/lpm_string_util.hpp"

#include <sstream>
#include <iomanip>

namespace Lpm {
namespace DFS {

TimestepParams::TimestepParams(const user::Input& input) {
  auto logger = lpm_logger();
  tfinal = input.get_option("tfinal").get_real();
  if (input.get_option("use_dt").get_bool()) {
    if (input.get_option("use_nsteps").get_bool()) {
      logger->error("TimestepParams error: cannot use both dt and nsteps to determine time step size; choose one.  Defaulting to dt.");
    }
    dt = input.get_option("dt").get_real();
    nsteps = int(tfinal / dt);
  }
  else {
    nsteps = input.get_option("nsteps").get_int();
    dt = tfinal / nsteps;
  }
}

Real TimestepParams::courant_number(const Real& dx, const Real& max_vel) const {
  return max_vel * dt / dx;
}

std::string TimestepParams::filename_piece() const {
  std::stringstream ss;
  ss << "tfinal" << std::setprecision(2) << tfinal << dt_str(dt) << "_";
  return ss.str();
}

}
}
