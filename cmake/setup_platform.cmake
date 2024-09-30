
macro(setup_platform)
 # do we have openmp?
 # TODO: Use Kokkos to determine this
if (APPLE)
   set(LPM_USE_OPENMP FALSE)
   find_package(Threads)
   if (Threads_FOUND)
     set(LPM_USE_THREADS TRUE)
     message(STATUS "threads enabled for Mac OS; setting LPM_USE_THREADS")
   endif()
else()
find_package(OpenMP)
if (OpenMP_FOUND)
   message(STATUS "setting LPM_USE_OPENMP")
   set(LPM_USE_OPENMP TRUE)
else()
  set(LPM_USE_OPENMP FALSE)
endif()
endif()

if (LPM_ENABLE_BOOST)
  find_package(Boost REQUIRED COMPONENTS math_tr1)
  set(LPM_USE_BOOST TRUE CACHE BOOL "Boost math enabled")
endif()

if (LPM_USE_OPENMP)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  message(STATUS "OpenMP is enabled")
endif()

if (Trilinos_DIR)
  include(lpm_find_trilinos)
else()
  message(STATUS "Trilinos_DIR not specified; looking for standalone TPLs")
  include(lpm_find_kokkos)
endif()

include(lpm_build_spdlog)
include(lpm_build_catch2)
include(lpm_find_vtk)
include(lpm_find_compose)
if (LPM_ENABLE_NETCDF)
  include(lpm_find_netcdf)
endif()
if (LPM_ENABLE_DFS)
  include(lpm_find_finufft)
  include(lpm_find_fftw3)
endif()
if (LPM_ENABLE_FastBVE)
  include(lpm_find_fastbve)
endif()

endmacro()
