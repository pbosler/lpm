set(KOKKOS_NEEDED FALSE CACHE LOGICAL "")

#https://github.com/E3SM-Project/E3SM/blob/master/components/homme/cmake/Kokkos.cmake

if (NOT DEFINED Kokkos_ROOT)
    set(KOKKOS_NEEDED TRUE)
    set(KOKKOS_SRC ${CMAKE_SOURCE_DIR}/tpl/kokkos)
    set(KOKKOS_BUILD_DIR ${CMAKE_BINARY_DIR}/kokkos)
    set(KOKKOS_INCLUDE_DIR ${KOKKOS_SRC}/core/src ${KOKKOS_BUILD_DIR})
    set(KOKKOS_LIBRARY_DIR ${KOKKOS_BUILD_DIR})
    Message(STATUS "Lpm will build Kokkos in ${KOKKOS_BUILD_DIR}")
else()
    message(STATUS "Lpm will use the Kokkos library installed at ${Kokkos_ROOT}")
    set(KOKKOS_INCLUDE_DIR ${Kokkos_ROOT}/include)
    set(KOKKOS_LIBRARY_DIR ${Kokkos_ROOT}/lib)
endif()


