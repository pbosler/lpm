#ifndef LPM_ASSERT_HPP
#define LPM_ASSERT_HPP

#include <assert.h>

#include <exception>
#include <sstream>

namespace Lpm {

/*
 * Asserts and error checking macros/functions.
 *
 * lpm_kernel* are for error checking within kokkos kernels.
 *
 * Any check with "assert" in the name is disabled for release builds
 *
 * For _msg checks, the msg argument can contain '<<' if not a kernel check.
 */

// Internal do not call directly
#define IMPL_THROW(condition, msg, exception_type)                      \
  do {                                                                  \
    if (!(condition)) {                                                 \
      std::stringstream _ss_;                                           \
      _ss_ << __FILE__ << ":" << __LINE__ << ": FAIL:\n" << #condition; \
      _ss_ << "\n" << msg;                                              \
      throw exception_type(_ss_.str());                                 \
    }                                                                   \
  } while (0)

#define LPM_IMPL_KERNEL_THROW(condition, msg)                              \
  do {                                                                 \
    if (!(condition)) {                                                \
      printf("KERNEL CHECK FAILED:\n   %s\n   %s\n", #condition, msg); \
      Kokkos::abort("");                                               \
    }                                                                  \
  } while (0)

#ifndef NDEBUG
#define LPM_ASSERT(condition) IMPL_THROW(condition, "", std::runtime_error)
#define LPM_ASSERT_MSG(condition, msg) \
  IMPL_THROW(condition, msg, std::runtime_error)
#define LPM_KERNEL_ASSERT(condition) LPM_IMPL_KERNEL_THROW(condition, "")
#define LPM_KERNEL_ASSERT_MSG(condition, msg) LPM_IMPL_KERNEL_THROW(condition, msg)
#else
#define LPM_ASSERT(condition) ((void)(0))
#define LPM_ASSERT_MSG(condition, msg) ((void)(0))
#define LPM_KERNEL_ASSERT(condition) ((void)(0))
#define LPM_KERNEL_ASSERT_MSG(condition, msg) ((void)(0))
#endif

#define LPM_REQUIRE(condition) IMPL_THROW(condition, "", std::runtime_error)
#define LPM_REQUIRE_MSG(condition, msg) \
  IMPL_THROW(condition, msg, std::runtime_error)
#define LPM_KERNEL_REQUIRE(condition) LPM_IMPL_KERNEL_THROW(condition, "")
#define LPM_KERNEL_REQUIRE_MSG(condition, msg) LPM_IMPL_KERNEL_THROW(condition, msg)
#define LPM_STOP(msg) IMPL_THROW(false, msg, std::runtime_error)
#define LPM_KERNEL_STOP(msg) LPM_IMPL_KERNEL_THROW(false, msg)

}  // namespace Lpm
#endif
