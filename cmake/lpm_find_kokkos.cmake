message(STATUS "LPM: looking for Kokkos at ${Kokkos_DIR}")
message(STATUS "LPM: looking for KokkosKernels at ${KokkosKernels_DIR}")
message(STATUS "LPM: looking for Compadre at ${Compadre_DIR}")

find_package(Kokkos REQUIRED HINTS ${Kokkos_DIR})

list(JOIN Kokkos_DEVICES ", " DEVICE_STR)
message(STATUS "LPM: Found Kokkos devices ${DEVICE_STR}")
if (${DEVICE_STR} MATCHES "CUDA")
  set(LPM_USE_CUDA TRUE CACHE BOOL "")
  message(STATUS "LPM: setting LPM_USE_CUDA = ${LPM_USE_CUDA}")
endif()

find_package(KokkosKernels REQUIRED HINTS ${KokkosKernels_DIR})

find_package(Compadre REQUIRED HINTS ${Compadre_DIR})

