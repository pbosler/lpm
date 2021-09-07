#ifndef LPM_FILENAME_UTIL_HPP
#define LPM_FILENAME_UTIL_HPP

#include "LpmConfig.h"
#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>

namespace Lpm {

template <typename SeedType>
struct BaseFilename {
  static constexpr int dt_digits = 5;

  static std::string counter_str(const Int count, const Int nfill = 4) {
    std::ostringstream ss;
    ss << "_" << std::setfill('0') << std::setw(nfill) << count;
    return ss.str();
  }

  static std::string vtk_suffix() {return ".vtp";}

  static std::string add_vtk_suffix(const std::string& basename) {
    return basename + vtk_suffix();
  }

  static std::string nc_suffix() {return ".nc";}

  static std::string add_netcdf_suffix(const std::string& basename) {
    return basename + nc_suffix();
  }

  static std::string dt_str(const Real dt) {
    std::ostringstream ss;
    ss << "_dt" << std::setprecision(dt_digits);
    return ss.str();
  }

  BaseFilename(const std::string& c, const int d) :
    case_name(c),
    init_depth(d) {
    base_str = case_name + "_" + SeedType::id() + std::to_string(init_depth);
  }

  BaseFilename(const std::string& c, const int d, const Real dt) :
    case_name(c),
    init_depth(d) {
    base_str = case_name + "_" + SeedType::id() + std::to_string(init_depth) + dt_str(dt);
  }

  std::string operator() () const {
    return base_str;
  }

  std::string operator() (const int counter) const {
    return base_str + counter_str(counter);
  }

  private:
    std::string case_name;
    int init_depth;
    std::string base_str;
};



} // namespace Lpm

#endif
