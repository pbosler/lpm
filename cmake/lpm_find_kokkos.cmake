message(STATUS "LPM: looking for Kokkos at ${Kokkos_DIR}")
message(STATUS "LPM: looking for KokkosKernels at ${KokkosKernels_DIR}")
message(STATUS "LPM: looking for Compadre at ${Compadre_DIR}")

find_package(Kokkos REQUIRED HINTS ${Kokkos_DIR})
if("OPENMP" IN_LIST Kokkos_DEVICES)
    message(STATUS "LPM: Kokkos was built with OpenMP support.")
    find_package(OpenMP REQUIRED)
    set(LPM_USE_OPENMP CACHE BOOL TRUE "")
else()
    message(STATUS "Kokkos OpenMP backend is NOT enabled.")
endif()

find_package(KokkosKernels REQUIRED HINTS ${KokkosKernels_DIR})

find_package(Compadre REQUIRED HINTS ${Compadre_DIR})

