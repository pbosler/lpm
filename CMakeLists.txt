cmake_minimum_required(VERSION 3.12)

set( LPM_CMAKE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake CACHE INTERNAL "")
list(APPEND CMAKE_MODULE_PATH
  ${LPM_CMAKE_PATH})

project (LPM CXX Fortran)

set(LPM_VERSION_MAJOR 3)
set(LPM_VERSION_MINOR 1)

set(CMAKE_CXX_STANDARD 14)
CMAKE_POLICY(SET CMP0074 NEW)
message(STATUS "Setting policy CMP0074 to use <Package>_ROOT environmental variables")

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

set(LPM_EXT_LIBRARIES ${CMAKE_DL_LIBS};${LPM_EXT_LIBRARIES})

printvar(LPM_EXT_LIBRARIES)
printvar(LPM_EXT_INCLUDE_DIRS)

#FIND_PACKAGE(VTK REQUIRED HINTS $ENV{VTK_ROOT})
#if (VTK_FOUND)
#    option (LPM_HAVE_VTK "Located VTK libraries." ON)
#    message ("VTK Found. Details:")
#    printvar(VTK_INCLUDE_DIRS)
#    printvar(VTK_LIBRARIES)
#endif()

#find_path(NETCDF_H_PATH "netcdf.h" HINTS $ENV{NETCDF_ROOT}/include)
#printvar(NETCDF_H_PATH)
#find_library(NETCDF_C netcdf HINTS ${NETCDF_H_PATH}/../lib)
#printvar(NETCDF_C)
#find_library(NETCDF_CXX netcdf_c++4 HINTS ${NETCDF_H_PATH}/../lib)
#printvar(NETCDF_CXX)
#
#set(HAS_NETCDF CACHE BOOL FALSE)
#if (NETCDF_H_PATH)
#  if (NETCDF_C)
#    if (NETCDF_CXX)
#      set(HAS_NETCDF TRUE)
#      option(LPM_HAVE_NETCDF "netcdf libraries found" ON)
#    endif()
#  endif()
#endif()
#printvar(HAS_NETCDF)



#find_package(Trilinos REQUIRED HINTS ${Trilinos_ROOT})
#if (Trilinos_FOUND)
#  message("Trilinos found. Details:")
#  printvar(Trilinos_INCLUDE_DIRS)
#  printvar(Trilinos_LIBRARIES)
#  printvar(Trilinos_TPL_INCLUDE_DIRS)
#  printvar(Trilinos_TPL_LIBRARIES)
#endif()
#
#find_package(CUDA QUIET)
#if (${CUDA_FOUND})
#  # Cuda found, but it's possible trilinos was not built with cuda support.
#  execute_process(COMMAND ${CMAKE_CXX_COMPILER} "--nvcc-wrapper-show"
#    RESULT_VARIABLE WRAPS_NVCC
#    OUTPUT_VARIABLE WRAPS_NVCC_OUT1
#    ERROR_QUIET)
#
#  if (DEFINED ENV{OMPI_CXX})
#    execute_process(COMMAND $ENV{OMPI_CXX} "--nvcc-wrapper-show"
#    RESULT_VARIABLE WRAPS_NVCC
#    OUTPUT_VARIABLE WRAPS_NVCC_OUT2
#    ERROR_QUIET)
#  endif()
#
#  string(FIND "${WRAPS_NVCC_OUT1} ${WRAPS_NVCC_OUT2}" "nvcc" pos)
#  if (${pos} GREATER -1)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")
#    option (LPM_HAVE_CUDA "LPM build with cuda support." ON)
#    message(STATUS "Cuda enabled!")
#  else ()
#    message(STATUS "Cuda was found, but the C++ compiler is not nvcc_wrapper.  Building without cuda.")
#  endif()
#endif()
#
#set(USE_SPHEREPACK CACHE BOOL FALSE)
#if (Spherepack_ROOT)
#    set(USE_SPHEREPACK TRUE)
#    option(LPM_HAVE_SPHEREPACK "Located Spherepack library." ON)
#    message("Looking for Spherepack at ${Spherepack_ROOT}")
#    ADD_LIBRARY(spherepack STATIC IMPORTED GLOBAL)
#    SET_TARGET_PROPERTIES(spherepack PROPERTIES IMPORTED_LOCATION ${Spherepack_ROOT}/lib/libspherepack.a)
#else()
#    Message(STATUS "Spherepack not found.")
#endif()

add_subdirectory(ext)

#ADD_SUBDIRECTORY (src)
#ADD_SUBDIRECTORY (tests)
#ADD_SUBDIRECTORY (examples)

CONFIGURE_FILE("${CMAKE_SOURCE_DIR}/LpmConfig.h.in" "${PROJECT_BINARY_DIR}/LpmConfig.h" @ONLY)

INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/src" "${PROJECT_BINARY_DIR}")# "${VTK_INCLUDE_DIRS}" "${Kokkos_ROOT}/include" "${NETCDF_H_PATH}")

ENABLE_TESTING()
