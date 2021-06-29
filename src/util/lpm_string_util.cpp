#include "util/lpm_string_util.hpp"
#include <sstream>

namespace Lpm {

std::string indent_string(const int tab_lev) {
  std::string result("");
  for (int i=0; i<tab_lev; ++i) {
    result += "\t";
  }
  return result;
}

std::string& tolower(std::string& s) {
  for (auto& c: s) {
    c = std::tolower(c);
  }
  return s;
}

std::string format_strings_as_list(const char** strings, const Short n) {
  std::stringstream ss;
  ss << "{";
  for (Short i=0; i<n-1; ++i) {
    ss << strings[i] << ", ";
  }
  ss << strings[n-1] << "}";
  return ss.str();
}


}
