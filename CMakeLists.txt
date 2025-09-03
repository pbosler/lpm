cmake_minimum_required(VERSION 3.12)

# Point CMake to our .cmake files
set( LPM_CMAKE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake CACHE INTERNAL "")
list(APPEND CMAKE_MODULE_PATH ${LPM_CMAKE_PATH})

project(LPM)
enable_language(CXX C Fortran)

# Set LPM version metadata
set(LPM_VERSION_MAJOR 3)
set(LPM_VERSION_MINOR 1)
set(LPM_VERSION_PATCH 0)

# Set c++ standard and env variable policy
set(CMAKE_CXX_STANDARD 17)
CMAKE_POLICY(SET CMP0074 NEW)
CMAKE_POLICY(SET CMP0144 NEW) # Requires CMake 3.27 or greater

set(CMAKE_CXX_EXTENSIONS OFF) # required by Kokkos

# Tools for CMake repetitive tasks & debugging
include(cmake_helpers)

find_package(MPI REQUIRED COMPONENTS C)
if (NOT MPI_C_FOUND)
  message(FATAL_ERROR "Error: cannot find MPI-C")
endif()

# Write basic configure info to console
message(STATUS "LPM: Configuring with build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "LPM: Generating project files in build directory: ${PROJECT_BINARY_DIR}")
message(STATUS "LPM: Looking for required third-party libraries: VTK, Spdlog, Catch2, Kokkos, KokkosKernels, Compose, Compadre")
set(LPM_USE_VTK TRUE CACHE BOOL "")
include(GNUInstallDirs)
if (Kokkos_DIR)
  find_package(Kokkos REQUIRED HINTS ${Kokkos_DIR})
else()
  message(FATAL_ERROR "Kokkos_DIR not specified.")
endif()

include(lpm_find_vtk)
include(lpm_find_spdlog_catch2)
include(lpm_find_kokkos)
include(lpm_find_compose)
if (LPM_ENABLE_DFS)
  set(DFS_INFO "LPM: Setting LPM_ENABLE_DFS=ON requires the Flatiron Institute's Nonuniform FFT library, finufft, which (currently, 03SEP2025) requires *both* single and double precision FFTW3 libraries. If these are not provided on your system (e.g., if `module load fftw3` doesn't include both single and double precision libraries), you will have to build them yourself.")
  message(STATUS ${DFS_INFO})
  include(lpm_find_fftw3_finufft)
  if (LPM_USE_FFTW3)
    message(STATUS "LPM: FFTW3 libraries enabled.")
  endif()
  if (LPM_USE_FINUFFT)
    message(STATUS "LPM: finnuft library enabled.")
  endif()
endif()
if (LPM_ENABLE_NETCDF)
  include(lpm_find_netcdf)
endif()

configure_file("${CMAKE_SOURCE_DIR}/LpmConfig.h.in" "${PROJECT_BINARY_DIR}/LpmConfig.h" @ONLY)

include(GNUInstallDirs)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(examples)

include(CTest)

