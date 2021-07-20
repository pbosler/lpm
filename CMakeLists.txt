cmake_minimum_required(VERSION 3.12)

# Point CMake to our .cmake files
set( LPM_CMAKE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake CACHE INTERNAL "")
list(APPEND CMAKE_MODULE_PATH ${LPM_CMAKE_PATH})

project (LPM CXX Fortran)

# Set LPM version metadata
set(LPM_VERSION_MAJOR 3)
set(LPM_VERSION_MINOR 1)
set(LPM_VERSION_PATCH 0)

# Set c++ standard and env variable policy
set(CMAKE_CXX_STANDARD 14)
CMAKE_POLICY(SET CMP0074 NEW)
message(STATUS "Setting policy CMP0074 to use <Package>_ROOT environmental variables")

# Tools for CMake repetitive tasks & debugging
include(cmake_helpers)

# Set all installation folders for third-party libraries, and figure out which
# ones have to be built.
include(setup_platform)
setup_platform()

# explicitly loading c++ features is necessary for some platforms
get_property(cxx_features GLOBAL PROPERTY CMAKE_CXX_KNOWN_FEATURES)
set(CMAKE_CXX_COMPILE_FEATURES ${cxx_features})
set(GLOBAL PROPERTY CMAKE_C_COMPILE_FEATURES ${cxx_features})

# handle MPI
#if (HAVE_MPI)
  find_package(MPI REQUIRED)
  if (NOT MPI_FOUND)
    message(FATAL_ERROR "Error: cannot find MPI")
  endif()
  set(LPM_USE_MPI ON)
#else()
#  message(STATUS "Building *without* MPI")
#  set(LPM_USE_MPI OFF)
#endif()

# handle OpenMP
if (LPM_USE_OPENMP)
  if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    message(STATUS "OpenMP is enabled")
  endif()
else()
  message(STATUS "OpenMP is disabled.")
endif()

# Write basic configure info to console
message(STATUS "Configuring for device: ${LPM_DEVICE} (${LPM_DEVICE_ARCH})")
message(STATUS "Configuring with build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Generating project files in build directory: ${PROJECT_BINARY_DIR}")

# Floating point precision
if (LPM_PRECISION STREQUAL "single")
  set(LPM_REAL_TYPE "float")
  set(LPM_DOUBLE_PRECISION 0)
elseif(LPM_PRECISION STREQUAL "double")
  set(LPM_REAL_TYPE "double")
  set(LPM_DOUBLE_PRECISION 1)
else()
  message(FATAL_ERROR "Invalid LPM_PRECISION: ${LPM_PRECISION} (use 'single' or 'double')")
endif()

# Build dependencies if necessary
add_subdirectory(ext)

# If we're building on a GPU, set the C++ compiler appropriately.
if (LPM_DEVICE STREQUAL CUDA)
  if (HAVE_MPI)
    set(ENV{OMPI_CXX_COMPILER} "${TRILINOS_LIBRARY_DIR}/bin/nvcc_wrapper")
  else()
    set(CMAKE_CXX_COMPILER "${TRILINOS_LIBRARY_DIR}/bin/nvcc_wrapper")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")
endif()

set(LPM_INCLUDE_DIRS ${LPM_INCLUDE_DIRS};${PROJECT_BINARY_DIR};${PROJECT_SOURCE_DIR}/src)

configure_file("${CMAKE_SOURCE_DIR}/LpmConfig.h.in" "${PROJECT_BINARY_DIR}/LpmConfig.h" @ONLY)

include(GNUInstallDirs)

add_subdirectory(src)
set(LPM_LIBRARIES lpm;${LPM_LIBRARIES})
add_subdirectory(tests)
add_subdirectory(examples)


printvar(LPM_LIBRARIES)

include(CTest)

