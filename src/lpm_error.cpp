#include "lpm_error.hpp"
#include "lpm_error_impl.hpp"
#include "lpm_assert.hpp"
#include <sstream>
#include <cmath>
#include <iomanip>

namespace Lpm {

std::vector<Real> convergence_rates(const std::vector<Real>& dx, const std::vector<Real>& ex) {
  LPM_REQUIRE(dx.size() == ex.size());
  std::vector<Real> result(dx.size(),0);
  for (int i=1; i<dx.size(); ++i) {
    result[i] = (log(ex[i]) - log(ex[i-1]))/(log(dx[i]) - log(dx[i-1]));
  }
  return result;
}

std::string convergence_table(const std::string dxlabel,
                              const std::vector<Real>& dx,
                              const std::string exlabel,
                              const std::vector<Real>& ex,
                              const std::vector<Real>& rate) {
  static constexpr Int col_width = 20;
  LPM_REQUIRE(dx.size() == ex.size());
  LPM_REQUIRE(dx.size() == rate.size());
  std::ostringstream ss;
  ss << "\n";
  ss << std::setw(col_width) << dxlabel
     << std::setw(col_width) << exlabel
     << std::setw(col_width) << "rate\n";
  ss << std::setw(col_width) << dx[0]
     << std::setw(col_width) << ex[0]
     << std::setw(col_width) << "---\n";
  for (int i=1; i<dx.size(); ++i) {
    ss << std::setw(col_width) << dx[i]
       << std::setw(col_width) << ex[i]
       << std::setw(col_width) << rate[i] << "\n";
  }
  return ss.str();
}

std::string ErrNorms::info_string(const std::string& label, const int tab_level) const {
    std::ostringstream ss;
    std::string tabstr;
    for (int i=0; i<tab_level; ++i) tabstr += "\t";
    ss << tabstr << label << " ErrNorms: l1 = " << l1 << " l2 = " << l2 << " linf = " << linf;
    return ss.str();
}

}
