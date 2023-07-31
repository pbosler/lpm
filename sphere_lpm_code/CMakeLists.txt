cmake_minimum_required(VERSION 3.12)

#
# Name your project
#
project(SphereFastPoisson)
enable_testing()



enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)

# set an odd cmake option that I find helpful
CMAKE_POLICY(SET CMP0074 NEW)


if (Kokkos_DIR)
  message(STATUS "looking for Kokkos at ${Kokkos_DIR}")
  find_package(Kokkos REQUIRED HINTS ${Kokkos_DIR})
else()
  message(FATAL_ERROR "Kokkos_DIR not specified; re-run CMake with -D Kokkos_DIR=<path-to-kokkos-install>")
endif()
if (KokkosKernels_DIR)
  message(STATUS "looking for KokkosKernels at ${KokkosKernels_DIR}")
  find_package(KokkosKernels REQUIRED HINTS ${KokkosKernels_DIR})
else()
  message(FATAL_ERROR "KokkosKernels_DIR not specified; re-run CMake with -D KokkosKernels_DIR=<path-to-kokkos-kernels-install>.")
endif()


## add_subdirectory(src)
## add_subdirectory(test)

##
## Handle FFTW3
##
if (FFTW3_DIR)
  set(FFTW3_INCLUDE_DIR ${FFTW3_DIR}/include)
  # to-do: pull appropriate (e.g., omp, threads) fftw based on kokkos configuration.
  set(FFTW3_LIBRARY ${FFTW3_DIR}/lib/libfftw3.a)
  if (NOT EXISTS ${FFTW3_INCLUDE_DIR})
    message(FATAL_ERROR "Could not find fftw3 include directory at ${FFTW3_INCLUDE_DIR}")
  endif ()
  if (NOT EXISTS ${FFTW3_LIBRARY})
    message(FATAL_ERROR "Could not find fftw3 library at ${FFTW3_LIBRARY}")
  else ()
    add_library(fftw3 STATIC IMPORTED GLOBAL)
    set_target_properties(fftw3 PROPERTIES IMPORTED_LOCATION ${FFTW3_LIBRARY})
  endif()
  else()
  message(FATAL_ERROR "FFTW3_DIR not specified.  Re-run CMake with -D FFTW3_DIR=<path-to-fftw3-install>")
 endif()
#
## HANDLE FFTW3 THREADS
if (FFTW3THREADS_DIR)
  set(FFTW3THREADS_INCLUDE_DIR ${FFTW3THREADS_DIR}/include)
#  # to-do: pull appropriate (e.g., omp, threads) fftw based on kokkos configuration.
  set(FFTW3THREADS_LIBRARY ${FFTW3THREADS_DIR}/lib/libfftw3_threads.a)
#
  if (NOT EXISTS ${FFTW3THREADS_INCLUDE_DIR})
    message(FATAL_ERROR "Could not find fftw3 include directory at ${FFTW3THREADS_INCLUDE_DIR}")
  endif ()
  if (NOT EXISTS ${FFTW3THREADS_LIBRARY})
    message(FATAL_ERROR "Could not find fftw3 library at ${FFTW3THREADS_LIBRARY}")
  else ()
    add_library(fftw3_threads STATIC IMPORTED GLOBAL)
    set_target_properties(fftw3_threads PROPERTIES IMPORTED_LOCATION ${FFTW3THREADS_LIBRARY})
  endif()
 else()
  message(FATAL_ERROR "FFTW3THREADS_DIR not specified.  Re-run CMake with -D FFTW3THREADS_DIR=<path-to-fftw3_threads-install>")
 endif()

 add_subdirectory(src)
 add_subdirectory(test)
 add_subdirectory(/Users/michaelchiware/finufft finufft)
 
#
#
## Handle OpenBlas
##
#if (OpenBlas_DIR)
#  set(OPENBLAS_INCLUDE_DIR ${OpenBlas_DIR}/include)
#  set(OPENBLAS_LIBRARY ${OpenBlas_DIR}/lib/libopenblas.a)
#  if (NOT EXISTS ${OPENBLAS_INCLUDE_DIR})
#    message(FATAL_ERROR "Could not find openblas include directory at ${OPENBLAS_INCLUDE_DIR}")
#  endif()
#  if (NOT EXISTS ${OPENBLAS_LIBRARY})
#    message(FATAL_ERROR "Could not find openblas library at ${OPENBLAS_LIBRARY}")
#  else()
#    add_library(openblas STATIC IMPORTED GLOBAL)
#    set_target_properties(openblas PROPERTIES IMPORTED_LOCATION ${OPENBLAS_LIBRARY})
#  endif()
#else()
#endif()
#
#add_executable(curl_test curltest.cpp)
#target_link_libraries(curl_test PUBLIC Kokkos::kokkos fftw3 fftw3_threads gomp openblas)
#target_include_directories(curl_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
#    ${CMAKE_CURRENT_SOURCE_DIR}/../cblas_fftw_version
#    ${FFTW3_INCLUDE_DIR}
#    ${FFTW3THREADS_INCLUDE_DIR}
#    ${OPENBLAS_INCLUDE_DIR})
