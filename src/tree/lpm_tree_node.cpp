#include "tree/lpm_tree_node.hpp"
#include "util/lpm_string_util.hpp"

namespace Lpm {
namespace tree {

std::string Node::info_string(const int tab_level) const {
  std::ostringstream ss;
  auto tabstr = indent_string(tab_level);
  ss << tabstr << "Node info:\n";
  tabstr += "\t";
  ss << tabstr << "n = " << n() << "\n";
  ss << tabstr << "level = " << level << "\n";
  ss << tabstr << "is_leaf() = " << std::boolalpha << is_leaf() << "\n";
  ss << tabstr << "parent = " << parent << "\n";
  ss << tabstr << "kids.size() = " << kids.size() << "\n";
  ss << tabstr << "box.volume() = " << box.volume() << "\n";
  return ss.str();
}

} // namespace tree
} // namespace Lpm
