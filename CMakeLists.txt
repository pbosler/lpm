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

include_directories("${PROJECT_BINARY_DIR}")
include_directories("${PROJECT_BINARY_DIR}/include")
include(GNUInstallDirs)
link_directories("${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(examples)

configure_file("${CMAKE_SOURCE_DIR}/LpmConfig.h.in" "${PROJECT_BINARY_DIR}/LpmConfig.h" @ONLY)

# Generate a blank lpm_version.cpp file
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/lpm_version.cpp
  COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/lpm_version.cpp
  COMMENT "Generating initial lpm_version.cpp ..."
)

# update lpm version info
add_custom_target(update_version_info ALL
  COMMAND ${BASH} ${PROJECT_SOURCE_DIR}/tools/update_version_info.sh ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/lpm_version.cpp
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/lpm_version.cpp
  COMMENT "Updating lpm version info ..."
)

#INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/src" "${PROJECT_BINARY_DIR}")# "${VTK_INCLUDE_DIRS}" "${Kokkos_ROOT}/include" "${NETCDF_H_PATH}")

include(CTest)
enable_testing()
