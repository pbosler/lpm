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


if (Kokkos_ROOT)
   message(STATUS "Lpm will use the Kokkos library installed at ${Kokkos_ROOT}")
   include(${Kokkos_ROOT}/kokkos_generated_settings.cmake)
#   set(KOKKOS_INCLUDE_DIRS ${Kokkos_ROOT}/include)
#   set(KOKKOS_LIBRARIES ${Kokkos_ROOT}/lib/libkokkos.a)
   set(KOKKOS_PREFIX ${Kokkos_ROOT})
   
   string(FIND "${KOKKOS_GMAKE_DEVICES}" "Cuda" cuda_str_pos)
   if (${cuda_str_pos} GREATER -1)
        Option(HAVE_CUDA "Kokkos/Cuda found." ON)    
   endif()
        
else()
    message(STATUS "Lpm will build Kokkos internally.  To use an external Kokkos installation, set the Kokkos_ROOT (case sensitive) CMake variable to its path.")
    set(KOKKOS_CMAKE_ARGS -DKOKKOS_ENABLE_DEPRECATED_CODE=OFF -DKOKKOS_ENABLE_SERIAL=ON -DKOKKOS_ENABLE_PTHREAD=OFF
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/tpl/kokkos -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_BUILD_TYPE=RelWithDebInfo)
    if (ENABLE_OPENMP)
        list(APPEND KOKKOS_CMAKE_ARGS -DKOKKOS_ENABLE_OPENMP=ON)
    else()
        list(APPEND KOKKOS_CMAKE_ARGS -DKOKKOS_ENABLE_OPENMP=OFF)
    endif()

    if (ENABLE_CUDA)
       FIND_PACKAGE(CUDA QUIET)
       if (${CUDA_FOUND})
           execute_process(COMMAND ${CMAKE_CXX_COMPILER} "--nvcc-wrapper-show" RESULT_VARIABLE WRAPS_NVCC
               OUTPUT_VARIABLE WRAPS_NVCC_OUT ERROR_QUIET)
           if (${WRAPS_NVCC} EQUAL 0)
               set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --expt-extended-lambda")
               OPTION(HAVE_CUDA "Kokkos will build with Cuda" ON)
               list(APPEND KOKKOS_CMAKE_ARGS -DKOKKOS_ENABLE_CUDA=ON -DKOKKOS_ENABLE_CUDA_LAMBDA -DKOKKOS_ARCH=${KOKKOS_ARCH_STR}) 
           else ()
               Message(WARNING "Cuda was found, but the C++ compiler is not nvcc_wrapper.  Building Kokkos without Cuda support.")
               set(ENABLE_CUDA FALSE CACHE LOGICAL FORCE "")
           endif()
       else()
          set(ENABLE_CUDA FALSE CACHE LOGICAL FORCE "")
          Message(WARNING "ENABLE_CUDA=ON but Cuda package not found.")
       endif()
    endif()
    include(kokkos.cmake)
#    set(KOKKOS_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/tpl/kokkos/include")
    set(KOKKOS_PREFIX "${CMAKE_BINARY_DIR}/tpl/kokkos")
endif()

ADD_LIBRARY(kokkos STATIC IMPORTED)
SET_TARGET_PROPERTIES(kokkos PROPERTIES IMPORTED_LOCATION ${KOKKOS_PREFIX}/lib/libkokkos.a)
INCLUDE_DIRECTORIES(${KOKKOS_PREFIX}/include)

if (Compadre_ROOT)
   message(STATUS "Lpm will use the Compadre GMLS toolkit installed at ${Compadre_ROOT}")
   set(COMPADRE_INCLUDE_DIRS "${Compadre_ROOT}/include")
   set(COMPADRE_LIBRARIES "${Compadre_ROOT}/lib/libcompadre.a")
else()
   message(STATUS "Lpm will build the Compadre GMLS toolkit internally. To use an external Compadre installation, set the Compadre_ROOT (case sensitive) CMake variable to its path.")
   set(COMPADRE_CMAKE_ARGS -DCompadre_USE_Pthread=OFF -DKokkosCore_PREFIX=${KOKKOS_PREFIX} -DCMAKE_BUILD_TYPE=RelWithDebInfo
       -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/tpl/compadre -DCMAKE_CXX_FLAGS=-fPIC -DBUILD_SHARED_LIBS=OFF)
   if (ENABLE_OPENMP)
       list(APPEND COMPADRE_CMAKE_ARGS -DCompadre_USE_OpenMP=ON)
   else()
       list(APPEND COMPADRE_CMAKE_ARGS -DCompadre_USE_OpenMP=OFF)
   endif()
   if (ENABLE_CUDA)
       list(APPEND COMPADRE_CMAKE_ARGS -DCompadre_USE_CUDA=ON)
   endif()
   include(compadre.cmake)
endif()

INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/src" 
                    "${PROJECT_BINARY_DIR}" 
                    "${VTK_INCLUDE_DIRS}" 
                    "${KOKKOS_PREFIX}/include"
                    #"${COMPADRE_INCLUDE_DIRS}"
                    )

get_property(cxx_features GLOBAL PROPERTY CMAKE_CXX_KNOWN_FEATURES)
set(CMAKE_CXX_COMPILE_FEATURES ${cxx_features})
set(GLOBAL PROPERTY CMAKE_C_COMPILE_FEATURES ${cxx_features})                    

ADD_SUBDIRECTORY (src)
ADD_SUBDIRECTORY (tests)
#ADD_SUBDIRECTORY (examples)

CONFIGURE_FILE("${CMAKE_SOURCE_DIR}/LpmConfig.h.in" "${PROJECT_BINARY_DIR}/LpmConfig.h" @ONLY)

ENABLE_TESTING()
