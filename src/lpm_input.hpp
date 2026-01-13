#ifndef LPM_INPUT_HPP
#define LPM_INPUT_HPP

#include <set>
#include <string>
#include <variant>

#include "LpmConfig.h"

namespace Lpm {
namespace user {

struct Option {
  typedef std::variant<Int, Real, std::string> variant_t;
  std::string name;
  std::string short_flag;
  std::string long_flag;
  std::string description;
  std::variant<Int, Real, bool, std::string> value;
  std::set<std::string> allowable_values;

  Option(const std::string& name, const std::string& sf, const std::string& lf,
         const std::string& desc, const Int default_value);

  Option(const std::string& name, const std::string& sf, const std::string& lf,
         const std::string& desc, const Real default_value);

  Option(const std::string& name, const std::string& sf, const std::string& lf,
         const std::string& desc, const bool default_value);

  Option(const std::string& name, const std::string& sf, const std::string& lf,
         const std::string& desc, const std::string& default_value);

  Option(const std::string& name, const std::string& sf, const std::string& lf,
         const std::string& desc, const std::string& default_value,
         const std::set<std::string>& allowable_values);

  std::string info_string(const int tab_level = 0) const;

  template <typename T>
  void set_value(const T& val) {
    value.emplace<T>(val);
    validate();
  }

  void validate() const;

  Int get_int() const;
  Real get_real() const;
  bool get_bool() const;
  std::string get_str() const;

  friend bool operator<(const Option& lhs, const Option& rhs) noexcept;
  friend bool operator>(const Option& lhs, const Option& rhs) noexcept;
  friend bool operator<=(const Option& lhs, const Option& rhs) noexcept;
  friend bool operator>=(const Option& lhs, const Option& rhs) noexcept;
  friend bool operator==(const Option& lhs, const Option& rhs) noexcept;
  friend bool operator!=(const Option& lhs, const Option& rhs) noexcept;
};

struct Input {
  std::string meta_description;
  std::map<std::string, Option> options;
  std::set<std::string> short_flags;
  std::set<std::string> long_flags;
  bool help_and_exit;

  Input() = default;
  explicit Input(const std::string& md);
  Input(const std::string& md, const std::vector<Option> opts);

  std::string usage() const;

  std::string info_string(const int tab_level = 0,
                          const bool verbose  = false) const;

  void add_option(const Option& default_opt);

  void parse_args(const int argc, char* argv[]);

  const Option& get_option(const std::string& name) const;
};

template <typename ProblemType>
Input setup_problem_input() {
  return Input();
}

}  // namespace user
}  // namespace Lpm

#endif
