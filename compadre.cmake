include(ExternalProject)

ExternalProject_Add(
    Compadre
    GIT_REPOSITORY "git@github.com:SNLComputation/compadre.git"
    GIT_TAG "kokkos_deprecated_fix"
    
    DEPENDS Kokkos
    
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    
    SOURCE_DIR "${CMAKE_SOURCE_DIR}/tpl/compadre"
    CMAKE_ARGS ${COMPADRE_CMAKE_ARGS}
    
    TEST_COMMAND ""   
)

set(COMPADRE_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/tpl/compadre/include")
set(COMPADRE_LIBRARIES "${CMAKE_BINARY_DIR}/tpm/compadre/lib/libcompadre.a")
#INCLUDE_DIRECTORIES(${COMPADRE_INCLUDE_DIRS})