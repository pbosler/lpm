#
# Trilinos: Required
#
message(STATUS "looking for Trilinos at ${Trilinos_DIR}")
list(APPEND CMAKE_MODULE_PATH ${Trilinos_DIR})
find_package(Trilinos CONFIG REQUIRED COMPONENTS ${Trilinos_COMPONENTS} PATHS ${Trilinos_DIR})
if (Trilinos_FOUND)
 message("\nTrilinos details:")
 message(STATUS "Trilinos_INCLUDE_DIRS = ${Trilinos_INCLUDE_DIRS}")
 message(STATUS "Trilinos_TPL_INCLUDE_DIRS = ${Trilinos_TPL_INCLUDE_DIRS}")
 message(STATUS "Trilinos_LIBRARY_DIRS = ${Trilinos_LIBRARY_DIRS}")
 message(STATUS "Trilinos_LIBRARIES = ${Trilinos_LIBRARIES}")
 message(STATUS "Trilinos_TPL_LIST = ${Trilinos_TPL_LIST}")
 message(STATUS "Trilinos_BUILD_SHARED_LIBS = ${Trilinos_BUILD_SHARED_LIBS}")
 message(STATUS "Trilinos_CXX_COMPILER_FLAGS = ${Trilinos_CXX_COMPILER_FLAGS}")
 message(STATUS "Trilinos_PACKAGE_LIST = ${Trilinos_PACKAGE_LIST}")
 message("\n")
else()
 message(FATAL_ERROR "ERROR: Trilinos not found.")
endif()

list(APPEND LPM_INCLUDE_DIRS ${Trilinos_INCLUDE_DIRS})
list(APPEND LPM_LIBRARIES ${Trilinos_LIBRARIES})
