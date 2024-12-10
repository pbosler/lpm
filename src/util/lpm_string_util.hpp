#ifndef LPM_STRING_UTIL_HPP
#define LPM_STRING_UTIL_HPP

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "LpmConfig.h"

namespace Lpm {

std::string indent_string(const int tab_lev);

std::string& tolower(std::string& s);

std::string format_strings_as_list(const char** strings, const Short n);

template <typename T>
static std::string sprarr(const std::string& name, const T* const v,
                          const size_t n) {
  std::ostringstream ss;
  ss << name << ": ";
  for (size_t i = 0; i < n; ++i) ss << " " << v[i];
  //   ss << "\n";
  return ss.str();
}

template <typename T>
static std::string sprarr(const std::vector<T>& v) {
  std::ostringstream ss;
  for (size_t i=0; i<v.size(); ++i) ss << " " << v[i];
  return ss.str();
}

inline std::string nc_suffix() { return ".nc"; }

template <typename T>
inline std::string zero_fill_str(const T ct, const Int nfill = 4) {
  static_assert(std::is_integral<T>::value, "integral type required");
  std::ostringstream ss;
  ss << std::setfill('0') << std::setw(nfill) << ct;
  return ss.str();
}

template <typename T>
inline std::string dt_str(const T val, const int width = 5) {
  static_assert(std::is_floating_point<T>::value,
                "floating point type required.");
  std::ostringstream ss;
  ss << "dt" << std::setprecision(width) << val;
  return ss.str();
}

template <typename T>
inline std::string float_str(const T val, const int width = 5) {
  static_assert(std::is_floating_point<T>::value,
                "floating point type required.");
  std::ostringstream ss;
  ss << std::setprecision(width) << val;
  return ss.str();
}

std::string rstrip(std::string str, const std::string& chars_to_strip=" \t\f\v\n\r");

std::string lstrip(std::string str, const std::string& chars_to_strip=" \t\f\v\n\r");

}  // namespace Lpm

#endif
