cmake_minimum_required(VERSION 3.1)

project (LPM)

set(LPM_VERSION_MAJOR 2)
set(LPM_VERSION_MINOR 5)

set(CMAKE_CXX_STANDARD 11)
CMAKE_POLICY(SET CMP0074 NEW)
MESSAGE(STATUS "Setting policy CMP0074 to use <Package>_ROOT environmental variables")

function (printvar var)
    message("${var}: ${${var}}")
endfunction()

function (list2str list str)
    string(REPLACE ";" " " tmp "${list}")
    set(${str} ${tmp} PARENT_SCOPE)
endfunction()

get_property(cxx_features GLOBAL PROPERTY CMAKE_CXX_KNOWN_FEATURES)
set(CMAKE_CXX_COMPILE_FEATURES ${cxx_features})
set(GLOBAL PROPERTY CMAKE_C_COMPILE_FEATURES ${cxx_features})

find_package(MPI REQUIRED)
FIND_PACKAGE(VTK REQUIRED HINTS $ENV{VTK_ROOT})
if (VTK_FOUND)
    option (LPM_HAVE_VTK "Located VTK libraries." ON)
    message ("VTK Found. Details:")
    printvar(VTK_INCLUDE_DIRS)
    printvar(VTK_LIBRARIES)
endif()

find_path(NETCDF_H_PATH "netcdf.h" HINTS $ENV{NETCDF_ROOT}/include)
printvar(NETCDF_H_PATH)
find_library(NETCDF_C netcdf HINTS ${NETCDF_H_PATH}/../lib)
printvar(NETCDF_C)
find_library(NETCDF_CXX netcdf_c++4 HINTS ${NETCDF_H_PATH}/../lib)
printvar(NETCDF_CXX)

if (LPM_ENABLE_DEBUG OR ${CMAKE_BUILD_TYPE} STREQUAL "DEBUG")
    option(LPM_ENABLE_DEBUG "Enable thorough debugging checks." ON)
endif()

FIND_PACKAGE(Trilinos REQUIRED HINTS ${Trilinos_ROOT})
if (Trilinos_FOUND)
  message("Trilinos found. Details:")
  printvar(Trilinos_INCLUDE_DIRS)
  printvar(Trilinos_LIBRARIES)
endif()

set(USE_SPHEREPACK CACHE BOOL FALSE)
if (Spherepack_ROOT)
    set(USE_SPHEREPACK TRUE)
    option(LPM_HAVE_SPHEREPACK "Located Spherepack library." ON)
    message("Looking for Spherepack at ${Spherepack_ROOT}")
    ADD_LIBRARY(spherepack STATIC IMPORTED GLOBAL)
    SET_TARGET_PROPERTIES(spherepack PROPERTIES IMPORTED_LOCATION ${Spherepack_ROOT}/lib/libspherepack.a)
else()
    Message(STATUS "Spherepack not found.")
endif()

#ADD_SUBDIRECTORY (src)
#ADD_SUBDIRECTORY (tests)
#ADD_SUBDIRECTORY (examples)

CONFIGURE_FILE("${CMAKE_SOURCE_DIR}/LpmConfig.h.in" "${PROJECT_BINARY_DIR}/LpmConfig.h" @ONLY)

INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/src" "${PROJECT_BINARY_DIR}" "${VTK_INCLUDE_DIRS}" "${Kokkos_ROOT}/include" "${NETCDF_H_PATH}")

ENABLE_TESTING()
