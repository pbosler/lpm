#ifndef LPM_STRING_UTIL_HPP
#define LPM_STRING_UTIL_HPP

#include "LpmConfig.h"
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>

namespace Lpm {

std::string indent_string(const int tab_lev);

std::string& tolower(std::string& s);

std::string format_strings_as_list(const char** strings, const Short n);

template<typename T>
static std::string sprarr (const std::string& name, const T* const v, const size_t n) {
  std::ostringstream ss;
  ss << name << ": ";
  for (size_t i = 0; i < n; ++i) ss << " " << v[i];
  ss << "\n";
  return ss.str();
}

std::string table_str(const std::vector<std::string>& headers, const std::vector<Real>& cols...);

} // namespace Lpm

#endif
