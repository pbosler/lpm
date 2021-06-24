
include(ExternalProject)
set(LPM_HDF5_SUBMODULE_PATH "" CACHE INTERNAL "")
get_filename_component(LPM_HDF5_SUBMODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/../ext/hdf5 ABSOLUTE)

define_property( GLOBAL
  PROPERTY LPM_HDF5_BUILT
  BRIEF_DOCS "True if hdf5 has already been built"
  FULL_DOCS "This property ensures that CMake only processes hdf4 once.")

get_property(IS_LPM_HDF5_BUILT GLOBAL PROPERTY LPM_HDF5_BUILT SET)

macro (LpmSetHdf5SourceDir)
  if (NOT HDF5_SOURCE_DIR)
    message(STATUS "HDF5_SOURCE_DIR not set; using submodule version.")
    set(HDF5_SOURCE_DIR ${LPM_HDF5_SUBMODULE_PATH} CACHE STRING "Hdf5 source directory")
  elseif( NOT EXISTS ${HDF5_SOURCE_DIR})
    message(FATAL_ERROR "Error: Please specify a valid source folder for Hdf5.\n   Path given: ${HDF5_SOURCE_DIR}")
  else()
    get_filename_component(ABS_HDF5_DIR ${HDF5_SOURCE_DIR} ABSOLUTE)
    if (ABS_HDF5_DIR STREQUAL LPM_HDF5_SUBMODULE_PATH)
      message(STATUS "Using Hdf5 in ${HDF5_SOURCE_DIR}\n      - User-supplied Hdf5 matches submodule path")
    else()
      message(STATUS "Using Hdf5 in ${HDF5_SOURCE_DIR}.\n    User-supplied HDF5 versions are not guaranteed to work.")
    endif()
  endif()
  set( HDF5_SOURCE_DIR "${HDF5_SOURCE_DIR}" CACHE STRING "Hdf5 source directory")
endmacro()

macro(BuildHdf5)
  if (NOT IS_LPM_HDF5_BUILT)
    LpmSetHdf5SourceDir()

    set(HDF5_BINARY_DIR ${CMAKE_BINARY_DIR}/ext/hdf5)

    set(HDF5_ENABLE_PARALLEL TRUE CACHE BOOL "use parallel hdf5")
    set(HDF5_BUILD_CPP_LIB FALSE CACHE BOOL "do not build c++ interface for hdf5")
    set(HDF5_BUILD_EXAMPLES FALSE CACHE BOOL "do not build hdf5 examples")
    set(HDF5_BUILD_HL_LIB TRUE CACHE BOOL "build the high level hdf5 library")
    set(HDF5_BUILD_TOOLS FALSE CACHE BOOL "do not build hdf5 build tools")
    add_subdirectory(${HDF5_SOURCE_DIR} ${HDF5_BINARY_DIR})

    set_property(GLOBAL PROPERTY LPM_HDF5_BUILT TRUE)
  endif()
endmacro()
