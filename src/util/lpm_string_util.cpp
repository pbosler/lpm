#include "util/lpm_string_util.hpp"

#include <sstream>

namespace Lpm {

std::string indent_string(const int tab_lev) {
  std::string result("");
  for (int i = 0; i < tab_lev; ++i) {
    result += "\t";
  }
  return result;
}

std::string& tolower(std::string& s) {
  for (auto& c : s) {
    c = std::tolower(c);
  }
  return s;
}

std::string format_strings_as_list(const char** strings, const Short n) {
  std::stringstream ss;
  ss << "{";
  for (Short i = 0; i < n - 1; ++i) {
    ss << strings[i] << ", ";
  }
  ss << strings[n - 1] << "}";
  return ss.str();
}

std::string rstrip(std::string str, const std::string& chars_to_strip) {
  std::string result(str);
  const auto found_stripchar = result.find_last_not_of(chars_to_strip);
  if (found_stripchar != std::string::npos) {
    result.erase(found_stripchar + 1);
  }
  else {
    result.clear();
  }
  return result;
}

std::string lstrip(std::string str, const std::string& chars_to_strip) {
  std::string result(str);
  const auto found_char = result.find_first_not_of(chars_to_strip);
  result.erase(0, std::min(found_char, result.size()-1));
  return result;
}

}  // namespace Lpm
