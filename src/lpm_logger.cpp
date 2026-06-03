#include "lpm_logger.hpp"

namespace Lpm {

std::shared_ptr<spdlog::logger> lpm_logger(const Log::level::level_enum lev) {
  std::shared_ptr<spdlog::logger> result = spdlog::get("lpm");
  if (!result) {
    result = spdlog::stdout_color_mt("lpm");
    result->set_level(lev);
  }
  return result;
}

}  // namespace Lpm
