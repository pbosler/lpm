cmake_minimum_required(VERSION 3.12)

# Point CMake to our .cmake files
set( LPM_CMAKE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake CACHE INTERNAL "")
list(APPEND CMAKE_MODULE_PATH ${LPM_CMAKE_PATH})

project (LPM CXX Fortran)

set(LPM_VERSION_MAJOR 3)
set(LPM_VERSION_MINOR 1)

set(CMAKE_CXX_STANDARD 14)
CMAKE_POLICY(SET CMP0074 NEW)
message(STATUS "Setting policy CMP0074 to use <Package>_ROOT environmental variables")

# Tools for CMake debugging
include(cmake_helpers)

# Set all installation folders for third-party libraries, and figure out which
# ones have to be built.
include(setup_platform)
setup_platform()

get_property(cxx_features GLOBAL PROPERTY CMAKE_CXX_KNOWN_FEATURES)
set(CMAKE_CXX_COMPILE_FEATURES ${cxx_features})
set(GLOBAL PROPERTY CMAKE_C_COMPILE_FEATURES ${cxx_features})

if (HAVE_MPI)
  find_package(MPI REQUIRED)
else()
  message(STATUS "Building *without* MPI")
endif()

add_subdirectory(ext)

#ADD_SUBDIRECTORY (src)
#ADD_SUBDIRECTORY (tests)
#ADD_SUBDIRECTORY (examples)

configure_file("${CMAKE_SOURCE_DIR}/LpmConfig.h.in" "${PROJECT_BINARY_DIR}/LpmConfig.h" @ONLY)

#INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/src" "${PROJECT_BINARY_DIR}")# "${VTK_INCLUDE_DIRS}" "${Kokkos_ROOT}/include" "${NETCDF_H_PATH}")

enable_testing()
