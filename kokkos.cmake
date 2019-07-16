include(ExternalProject)

ExternalProject_Add(
    Kokkos
    GIT_REPOSITORY "git@github.com:kokkos/kokkos.git"
    GIT_TAG "develop"
    
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    
    SOURCE_DIR "${CMAKE_SOURCE_DIR}/tpl/kokkos"
    CMAKE_ARGS ${KOKKOS_CMAKE_ARGS}
    
    TEST_COMMAND ""
)

set(KOKKOS_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/tpl/kokkos/include")
set(KOKKOS_LIBRARIES "${CMAKE_BINARY_DIR}/tpl/kokkos/lib/libkokkos.a")
INCLUDE_DIRECTORIES(${KOKKOS_INCLUDE_DIRS})


