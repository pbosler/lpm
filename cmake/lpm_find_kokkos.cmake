message(STATUS "LPM: looking for Kokkos at ${Kokkos_DIR}")
message(STATUS "LPM: looking for KokkosKernels at ${KokkosKernels_DIR}")
message(STATUS "LPM: looking for Compadre at ${Compadre_DIR}")

find_package(Kokkos REQUIRED HINTS ${Kokkos_DIR})

find_package(KokkosKernels REQUIRED HINTS ${KokkosKernels_DIR})

find_package(Compadre REQUIRED HINTS ${Compadre_DIR})

