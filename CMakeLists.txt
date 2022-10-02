cmake_minimum_required(VERSION 3.12)

# Point CMake to our .cmake files
set( LPM_CMAKE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake CACHE INTERNAL "")
list(APPEND CMAKE_MODULE_PATH ${LPM_CMAKE_PATH})

project(LPM)
enable_language(CXX)

# Set LPM version metadata
set(LPM_VERSION_MAJOR 3)
set(LPM_VERSION_MINOR 1)
set(LPM_VERSION_PATCH 0)

# Set c++ standard and env variable policy
set(CMAKE_CXX_STANDARD 17)
CMAKE_POLICY(SET CMP0074 NEW)
message(STATUS "Setting policy CMP0074 to use <Package>_ROOT environmental variables")
set(CMAKE_CXX_EXTENSIONS OFF)

# Tools for CMake repetitive tasks & debugging
include(cmake_helpers)

# explicitly loading c++ features is necessary for some platforms
get_property(cxx_features GLOBAL PROPERTY CMAKE_CXX_KNOWN_FEATURES)
set(CMAKE_CXX_COMPILE_FEATURES ${cxx_features})
set(GLOBAL PROPERTY CMAKE_C_COMPILE_FEATURES ${cxx_features})


find_package(MPI REQUIRED COMPONENTS C)
if (NOT MPI_C_FOUND)
  message(FATAL_ERROR "Error: cannot find MPI-C")
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

# Add external libraries in-tree
set(LPM_EXT_LIBRARIES ${CMAKE_DL_LIBS} ${MPI_C_LIBRARIES} )
set(LPM_EXT_INCLUDE_DIRS ${MPI_C_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/ext/catch2/include)
#add_subdirectory(ext)
# Set all installation folders for third-party libraries, and figure out which
# ones have to be built.
include(setup_platform)
setup_platform()
message("LPM_EXT_INCLUDE_DIRS = ${LPM_EXT_INCLUDE_DIRS}")
message("LPM_EXT_LIBRARIES = ${LPM_EXT_LIBRARIES}")

# If we're building on a GPU, set the C++ compiler appropriately.
if (LPM_DEVICE STREQUAL CUDA)
  set(ENV{OMPI_CXX_COMPILER} "${TRILINOS_LIBRARY_DIR}/bin/nvcc_wrapper")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")
endif()

set(LPM_INCLUDE_DIRS ${LPM_INCLUDE_DIRS}
                     ${PROJECT_BINARY_DIR}
                     ${PROJECT_SOURCE_DIR}/src
                     ${MPI_INCLUDE_DIRS}
                     ${LPM_EXT_INCLUDE_DIRS}
                     )

set(LPM_LIBRARIES ${LPM_EXT_LIBRARIES})

configure_file("${CMAKE_SOURCE_DIR}/LpmConfig.h.in" "${PROJECT_BINARY_DIR}/LpmConfig.h" @ONLY)

include(GNUInstallDirs)

printvar(LPM_INCLUDE_DIRS)
printvar(LPM_LIBRARIES)


add_subdirectory(src)
set(LPM_LIBRARIES lpm
                  ${LPM_LIBRARIES})
add_subdirectory(tests)
add_subdirectory(examples)



include(CTest)

