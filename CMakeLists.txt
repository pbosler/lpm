cmake_minimum_required(VERSION 3.1)

project (LPM)

set(LPM_VERSION_MAJOR 2)
set(LPM_VERSION_MINOR 5)

set(CMAKE_CXX_STANDARD 11)

function (printvar var)
    message("${var}: ${${var}}")
endfunction()

function (list2str list str)
    string(REPLACE ";" " " tmp "${list}")
    set(${str} ${tmp} PARENT_SCOPE)
endfunction()

find_package(MPI REQUIRED)
FIND_PACKAGE(VTK REQUIRED HINTS $ENV{VTK_ROOT})
if (VTK_FOUND)
    option (HAVE_VTK "Located VTK libraries." ON)
    message ("VTK Found. Details:")
    printvar(VTK_INCLUDE_DIRS)
    printvar(VTK_LIBRARIES)
endif()

set(CUDA_BUILD FALSE)
if (Kokkos_ROOT)
    option(HAVE_KOKKOS "Located Kokkos library." ON)
    message("Kokkos found.")
    include(${Kokkos_ROOT}/kokkos_generated_settings.cmake)
    INCLUDE_DIRECTORIES(${Kokkos_ROOT}/include)
    ADD_LIBRARY(kokkos STATIC IMPORTED)
    SET_TARGET_PROPERTIES(kokkos PROPERTIES IMPORTED_LOCATION ${Kokkos_ROOT}/lib/libkokkos.a)
#    set(Kokkos_INCLUDE_DIRS ${Kokkos_ROOT}/include)
#    set(Kokkos_LIBRARIES ${Kokkos_ROOT}/lib/libkokkos.a)
#    
#    string(FIND "${KOKKOS_GMAKE_DEVICES}" "Cuda" cuda_str_pos)
#    if (${cuda_str_pos} GREATER -1)
#        option(HAVE_CUDA "If true, indicates a Kokkos Cuda build." ON)
#        set(CUDA_BUILD TRUE)
#        execute_process(COMMAND ${CMAKE_CXX_COMPILER} "--nvcc-wrapper-show" 
#	        RESULT_VARIABLE WRAPS_NVCC OUTPUT_VARIABLE WRAPS_NVCC_OUT)
#        string (FIND ${WRAPS_NVCC_OUT} "nvcc" pos)
#    endif()
else()
    Message(FATAL_ERROR "Kokkos is required. Set CMake variable Kokkos_ROOT.")
endif()

if (Compadre_ROOT)
    option(HAVE_COMPADRE "Located Compadre GMLS toolkit" ON)
    message("Compadre located. ")
    INCLUDE_DIRECTORIES(${Compadre_ROOT}/include)
    ADD_LIBRARY(compadre STATIC IMPORTED)
    SET_TARGET_PROPERTIES(compadre PROPERTIES IMPORTED_LOCATION ${Compadre_ROOT}/lib/libcompadre.a)
else ()
    Message(STATUS "Compadre GMLS toolkit not found.")
endif() 

list2str("${KOKKOS_CXX_FLAGS}" Kokkos_CXXFLAGS)
list2str("${KOKKOS_LD_FLAGS}" Kokkos_LDFLAGS)
set (LPM_CXXFLAGS "${Kokkos_CXXFLAGS} ${CMAKE_CXX_FLAGS}")
if (CUDA_BUILD)
    set( LPM_CXXFLAGS "${LPM_CXXFLAGS} --expt-relaxed-constexpr")
endif()
set(LPM_LDFLAGS "-L/lib64 -ldl")

printvar(LPM_CXXFLAGS)
printvar(LPM_LDFLAGS)

ADD_SUBDIRECTORY (src)
ADD_SUBDIRECTORY (tests)
#ADD_SUBDIRECTORY (examples)

CONFIGURE_FILE("${CMAKE_SOURCE_DIR}/LpmConfig.h.in" "${PROJECT_BINARY_DIR}/LpmConfig.h" @ONLY)

INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/src" "${PROJECT_BINARY_DIR}" "${VTK_INCLUDE_DIRS}" "${Kokkos_ROOT}/include")

ENABLE_TESTING()
