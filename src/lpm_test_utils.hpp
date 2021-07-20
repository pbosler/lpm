#ifndef LPM_TEST_UTILS_HPP
#define LPM_TEST_UTILS_HPP

#include <map>

namespace Lpm {

/// this struct ensures there is exactly 1 "session" for use with Catch2
struct TestSession {
  static TestSession& get() {
    static TestSession s;
    return s;
  }

  std::map<std::string,std::string> params;

  private:
    TestSession() = default;
};

// This routine tries to detect the device id that the current MPI rank
// should use during a test.
// The routine relies on CTest RESOURCE_GROUPS feature. In particular,
// IF one specify resource groups properties when adding the test during
// cmake configuration, AND IF one passes a resource specification file
// to ctest, THEN ctest sets some env variables, which the test can read
// to figure out the resources its running on.
//
// For more info, see
//    https://cmake.org/cmake/help/latest/prop_test/RESOURCE_GROUPS.html
//    https://cmake.org/cmake/help/latest/manual/ctest.1.html#ctest-resource-environment-variables
//
// Note 1: the name of resources is completely meaningless to ctest.
//         Here we use 'devices', but it's not necessary.
// Note 2: CTest uses the resources specs also to schedule tests execution:
//         knowing the amount of resource available (through the spec file),
//         as well as the resources requested by each test, it can make sure
//         all resources are used, without oversubscribing them.
// Note 3: This feature is only available since CMake 3.16. However, the cmake
//         logic is completely innocuous with older cmake versions, and since
//         older versions do not set any CTEST_XYZ env variable, this function
//         will simply return -1 also on CUDA, which means that Kokkos will
//         autonomously decide which device to use on GPU (usually, the first one).
// Note 4: it only does something on CUDA builds; with every other
//         kokkos device, it returns -1.
int get_test_device (const int mpi_rank);

} // namespace Lpm
#endif
