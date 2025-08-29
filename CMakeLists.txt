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
# message(STATUS "Setting policy CMP0074 and CMP0144 to use <Package>_ROOT environmental variables")

set(CMAKE_CXX_EXTENSIONS OFF) # required by Kokkos

# Tools for CMake repetitive tasks & debugging
include(cmake_helpers)

# explicitly loading c++ features is necessary for some platforms
#get_property(cxx_features GLOBAL PROPERTY CMAKE_CXX_KNOWN_FEATURES)
#set(CMAKE_CXX_COMPILE_FEATURES ${cxx_features})
#set(GLOBAL PROPERTY CMAKE_C_COMPILE_FEATURES ${cxx_features})

find_package(MPI REQUIRED COMPONENTS C)
if (NOT MPI_C_FOUND)
  message(FATAL_ERROR "Error: cannot find MPI-C")
endif()

# Write basic configure info to console
message(STATUS "Configuring with build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Generating project files in build directory: ${PROJECT_BINARY_DIR}")
message(STATUS "Looking for required third-party libraries: VTK, Spdlog, Catch2, Kokkos, KokkosKernels, Compose, Compadre")
include(GNUInstallDirs)
if (Kokkos_DIR)
  message(STATUS "looking for Kokkos at ${Kokkos_DIR}")
  find_package(Kokkos REQUIRED HINTS ${Kokkos_DIR})
else()
  message(FATAL_ERROR "Kokkos_DIR not specified.")
endif()

include(lpm_find_vtk)
include(lpm_find_spdlog_catch2)
include(lpm_find_kokkos)
include(lpm_find_compose)

# Add external libraries
#set(LPM_EXT_LIBRARIES ${CMAKE_DL_LIBS} ${MPI_C_LIBRARIES} )
#set(LPM_EXT_INCLUDE_DIRS ${MPI_C_INCLUDE_DIRS} )
# Set all installation folders for third-party libraries, and figure out which
# ones have to be built.
#message(STATUS "setting up platform")
#include(setup_platform)
#setup_platform()
#message("LPM_EXT_INCLUDE_DIRS = ${LPM_EXT_INCLUDE_DIRS}")
#message("LPM_EXT_LIBRARIES = ${LPM_EXT_LIBRARIES}")

set(LPM_INCLUDE_DIRS ${PROJECT_BINARY_DIR}
                     ${PROJECT_SOURCE_DIR}/src
                    )

#set(LPM_LIBRARIES ${LPM_EXT_LIBRARIES})

configure_file("${CMAKE_SOURCE_DIR}/LpmConfig.h.in" "${PROJECT_BINARY_DIR}/LpmConfig.h" @ONLY)

include(GNUInstallDirs)

printvar(LPM_INCLUDE_DIRS)

add_subdirectory(src)
set(LPM_LIBRARIES lpm
                  lpm_fortran
                  ${LPM_LIBRARIES})

printvar(LPM_LIBRARIES)

add_subdirectory(tests)
add_subdirectory(examples)



include(CTest)

