#ifndef LPM_STL_UTILS_HPP
#define LPM_STL_UTILS_HPP

#include "LpmConfig.h"

namespace Lpm {

/// This can go away once c++20 comes in
template <typename map_type, typename key_type>
bool map_contains(const map_type& map, const key_type& key) {
  const auto iter = map.find(key);
  return (iter != map.end());
}

} // namespace Lpm

#endif
