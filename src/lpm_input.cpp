#include "lpm_assert.hpp"
#include "lpm_input.hpp"
#include "util/lpm_string_util.hpp"
#include <iostream>
#include <sstream>

namespace Lpm {
namespace user {

Option::Option(const std::string& name, const std::string& sf, const std::string& lf, const std::string& desc,
    const Int default_value) :
    name(name),
    short_flag(sf),
    long_flag(lf),
    description(desc),
    value{default_value},
    allowable_values() {}

Option::Option(const std::string& name, const std::string& sf, const std::string& lf, const std::string& desc,
    const double default_value) :
    name(name),
    short_flag(sf),
    long_flag(lf),
    description(desc),
    value{default_value},
    allowable_values() {}

Option::Option(const std::string& name, const std::string& sf, const std::string& lf, const std::string& desc,
    const bool default_value) :
    name(name),
    short_flag(sf),
    long_flag(lf),
    description(desc),
    value{default_value} {}

Option::Option(const std::string& name, const std::string& sf, const std::string& lf, const std::string& desc,
    const std::string& default_value) :
    name(name),
    short_flag(sf),
    long_flag(lf),
    description(desc),
    value{default_value} {}

Option::Option(const std::string& name, const std::string& sf, const std::string& lf, const std::string& desc,
    const std::string& default_value,
    const std::set<std::string>& allowable_values) :
    name(name),
    short_flag(sf),
    long_flag(lf),
    description(desc),
    value{default_value},
    allowable_values(std::move(allowable_values)) {
        LPM_REQUIRE_MSG(allowable_values.count(default_value) == 1,
          "Option: default value must be included in allowable values");
    }

std::string Option::info_string(const int tab_level) const {
  std::string tab_str = indent_string(tab_level);
  std::ostringstream ss;
  const std::string separator = "  ";
  ss << tab_str << "Option: " << name << " (" << description << ")";
  ss << separator;
  ss << "[" << short_flag << ", " << long_flag << "]" << separator;
  if (std::holds_alternative<Int>(value)) {
    ss << tab_str << "value (Int) = " << this->get_int();
  }
  else if (std::holds_alternative<Real>(value))  {
    ss << tab_str << "value (Real) = " << this->get_real();
  }
  else if (std::holds_alternative<std::string>(value) ) {
    ss << tab_str << "value (std::string) = " << this->get_str();
    if (!allowable_values.empty()) {
      ss << "\n allowable_values: ";
      for (const auto& s : allowable_values) {
        ss << s << " ";
      }
    }
  }
  else if (std::holds_alternative<bool>(value) ) {
    ss << tab_str << "value (bool) = " << std::boolalpha << this->get_bool();
  }
  return ss.str();
}

Int Option::get_int() const {
  Int val;
  try {
    val = std::get<Int>(value);
  }
  catch (const std::bad_variant_access& ex) {
    std::ostringstream ss;
    ss << "Option: " << description << " bad int access. " << ex.what();
    LPM_STOP(ss.str());
  }
  return val;
}

bool Option::get_bool() const {
  bool val;
  try {
    val = std::get<bool>(value);
  }
  catch (const std::bad_variant_access& ex) {
    std::ostringstream ss;
    ss << "Option: " << description << " bad bool access. " << ex.what();
    LPM_STOP(ss.str());
  }
  return val;
}

Real Option::get_real() const {
  Real val;
  try {
    val = std::get<Real>(value);
  }
  catch (const std::bad_variant_access& ex) {
    std::ostringstream ss;
    ss << "Option: " << description << " bad real access. " << ex.what();
    LPM_STOP(ss.str());
  }
  return val;
}

std::string Option::get_str() const {
  std::string val;
  try {
    val = std::get<std::string>(value);
  }
  catch (const std::bad_variant_access& ex) {
    std::ostringstream ss;
    ss << "Option: " << description << " bad string access. " << ex.what();
    LPM_STOP(ss.str());
  }
  return val;
}

void Option::validate() const {
  if (std::holds_alternative<std::string>(value)) {
    if (!allowable_values.empty()) {
      const auto val = get_str();
      const bool is_valid = (allowable_values.count(val) == 1);
      if (!is_valid) {
        std::ostringstream ss;
        ss << "Option: " << description << " has invalid value: " << val << "\n";
        ss << "   allowable values are: ";
        for (const auto& s : allowable_values) {
          ss << s << " ";
        }
        LPM_STOP(ss.str());
      }
    }
  }
}


bool operator < (const Option& lhs, const Option& rhs) noexcept {
  return std::tie(lhs.name, lhs.short_flag, lhs.long_flag, lhs.description, lhs.value) <
         std::tie(rhs.name, rhs.short_flag, rhs.long_flag, rhs.description, rhs.value);
}

bool operator == (const Option& lhs, const Option& rhs) noexcept {
  return std::tie(lhs.name, lhs.short_flag, lhs.long_flag, lhs.description, lhs.value) ==
         std::tie(rhs.name, rhs.short_flag, rhs.long_flag, rhs.description, rhs.value);
}

bool operator != (const Option& lhs, const Option& rhs) noexcept {
  return !(lhs == rhs);
}

bool operator >= (const Option& lhs, const Option& rhs) noexcept {
  return !(lhs < rhs);
}

bool operator <= (const Option& lhs, const Option& rhs) noexcept {
  return (lhs < rhs) or (lhs == rhs);
}

bool operator > (const Option& lhs, const Option& rhs) noexcept {
  return !(lhs <= rhs);
}



Input::Input(const std::string& md) : meta_description(md), help_and_exit(false) {}

Input::Input(const std::string& md, const std::vector<Option> opts) : meta_description(md), help_and_exit(false) {
  for (int i=0; i<opts.size(); ++i) {
    add_option(opts[i]);
  }
}

std::string Input::usage() const {
  std::ostringstream ss;
  ss << meta_description << " " << "Input / usage:\n";
  constexpr Int tabs = 1;
  for (const auto& o : options) {
    ss << o.second.info_string(tabs);
  }
  return ss.str();
}

std::string Input::info_string(const int tab_level, const bool verbose) const  {
  std::ostringstream ss;
  std::string tab_str = indent_string(tab_level);
  ss << tab_str << "Input info:\n";
  ss << tab_str << "meta: " << meta_description << "\n";
  for (const auto& o : options) {
    ss << o.second.info_string(tab_level) << "\n";
  }
  return ss.str();
}

void Input::add_option(const Option& default_opt) {
  if (short_flags.count(default_opt.short_flag) > 0) {
    std::ostringstream ss;
    ss << "Input::add_option error: Option short flags must be unique ("
       << default_opt.short_flag << ")";
    LPM_STOP(ss.str());
  }
  if (long_flags.count(default_opt.long_flag) > 0) {
    std::ostringstream ss;
    ss << "Input::add_option error: Option long flags must be unique ("
       << default_opt.long_flag << ")";
    LPM_STOP(ss.str());
  }
  short_flags.insert(default_opt.short_flag);
  long_flags.insert(default_opt.long_flag);
  options.emplace(default_opt.name, std::move(default_opt));
}

void Input::parse_args(const int argc, char* argv[]) {
  std::cout << "parsing " << argc << " args\n";
  for (int i=1; i<argc; ++i) {
    const std::string& token = argv[i];
      if (token == "-h" or token == "--help") {
        help_and_exit = true;
      break;
    }
    for (auto&& [name, opt] : options) {
      if (opt.short_flag == token or opt.long_flag == token) {
        if (std::holds_alternative<Int>(opt.value)) {
          opt.value = std::stoi(argv[++i]);
        }
        else if (std::holds_alternative<Real>(opt.value)) {
          opt.value = std::stod(argv[++i]);
        }
        else if (std::holds_alternative<bool>(opt.value)) {
          opt.value = true;
        }
        else if (std::holds_alternative<std::string>(opt.value)) {
          opt.value = std::string(argv[++i]);
        }
        else {
          LPM_STOP("Input::parse_args error: bad variant access.");
        }
      }
    }
  }
}

const Option& Input::get_option(const std::string& name) const {
  return options.at(name);
}

} // namespace user
} // namespace Lpm
