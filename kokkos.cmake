include(ExternalProject)

ExternalProject_Add(
    Kokkos
    GIT_REPOSITORY "git@github.com:kokkos/kokkos.git"
    GIT_TAG "develop"
    
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    
    PREFIX=${CMAKE_BINARY_DIR}/tpl/kokkos
    
    BUILD_ALWAYS ON
    
    SOURCE_DIR "${CMAKE_SOURCE_DIR}/tpl/kokkos"
    CMAKE_ARGS ${KOKKOS_CMAKE_ARGS}
    
    TEST_COMMAND ""
)


