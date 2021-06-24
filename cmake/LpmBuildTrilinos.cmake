set (LPM_TRILINOS_SUBMODULE_PATH "" CACHE INTERNAL "")
get_filename_component(LPM_TRILINOS_SUBMODULE_PATH
${CMAKE_CURRENT_LIST_DIR}/../ext/trilinos ABSOLUTE)

define_property(GLOBAL
  PROPERTY LPM_TRILINOS_BUILT
  BRIEF_DOCS "True if trilinos has already been processed"
  FULL_DOCS "This property ensures that CMake only processes the Trilinos subdirectory once")

get_property(IS_LPM_TRILINOS_BUILT GLOBAL PROPERTY LPM_TRILINOS_BUILT SET)

macro (LpmSetTrilinosSourceDir)
  if (NOT Trilinos_SOURCE_DIR)
    message(STATUS "Trilinos_SOURCE_DIR not specified; using submodule version.")
    set( Trilinos_SOURCE_DIR ${LPM_TRILINOS_SUBMODULE_PATH} CACHE STRING "Trilinos source directory")

  elseif(NOT EXISTS ${Trilinos_SOURCE_DIR})
    message(FATAL_ERROR "Error: Please specify a valid source folder for Trilinos.\n     Provided path: ${Trilinos_SOURCE_DIR}")
  else()
    get_filename_component(ABS_TRILINOS_DIR ${Trilinos_SOURCE_DIR} ABSOLUTE)
    if (ABS_TRILINOS_DIR STREQUAL LPM_TRILINOS_SUBMODULE_PATH)
      message(STATUS "Using Trilinos in ${Trilinos_SOURCE_DIR}\n    - User-supplied Trilinos path matches submodule path")
    else()
      message(STATUS "Using Trilinos in ${Trilinos_SOURCE_DIR}.\n    User-supplied Trilinos versions are not guaranteed to work.")
    endif()
  endif()
  # If the variable existed, but not in the cache, set it in the cache
  set (Trilinos_SOURCE_DIR "${Trilinos_SOURCE_DIR}" CACHE STRING "Trilinos source directory")
endmacro()

# Process Trilinos
macro(BuildTrilinos)
  if (NOT IS_LPM_TRILINOS_BUILT)
    LpmSetTrilinosSourceDir()
    # set trilinos' build/install location
    set(Trilinos_BINARY_DIR ${CMAKE_BINARY_DIR}/ext/trilinos)

    if (CMAKE_BUILD_TYPE STREQUAL DEBUG)
      set( Kokkos_ENABLE_DEBUG TRUE CACHE BOOL "Enable Kokkos debug")
    endif()
    set( Trilinos_ENABLE_Fortran FALSE CACHE BOOL "Disable trilinos fortran")
    set( TPL_ENABLE_MPI TRUE CACHE BOOL "Trilinos enable mpi")
    if ( Kokkos_ENABLE_OPENMP)
      set(Trilinos_ENABLE_OPENMP TRUE CACHE BOOL "Trilinos enable openmp")
    endif()
    if ( Kokkos_ENABLE_CUDA)
      set(TPL_ENABLE_CUDA TRUE CACHE BOOL "Trilinos enable cuda")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -lineinfo -Xcudafe --diag_suppress=conversion_function_not_usable -Xcudafe --diag_suppress=cc_clobber_ignored -Xcudafe --diag_suppress=code_is_unreachable")
      set(TPL_ENABLE_CUBLAS TRUE CACHE BOOL)
      set(TPL_ENABLE_CUSOLVER TRUE CACHE BOOL)
    endif()
    set(Trilinos_ENABLE_ALL_PACKAGES FALSE CACHE BOOL "Trilinos disable all packages")
    set(Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES FALSE CACHE BOOL "Trilinos disable optional packages")
    set(Trilinos_ENABLE_TEUCHOS TRUE CACHE BOOL "Enable Teuchos")
    set(Trilinos_ENABLE_Kokkos TRUE CACHE BOOL "Enable Kokkos")
    set(Trilinos_ENABLE_KokkosCore TRUE CACHE BOOL "Enable KokkosCore")
    set(Trilinos_ENABLE_KokkosAlgorithms TRUE CACHE BOOL "Enable KokkosAlgorithms")
    set(Trilinos_ENABLE_KokkosKernels TRUE CACHE BOOL "Enable KokkosKernels")
    set(Kokkos_ENABLE_PROFILING TRUE CACHE BOOL "Enable kernel profiling")
    set(Kokkos_ENABLE_SERIAL TRUE CACHE BOOL "Enable serial node type")
    set(Trilinos_ENABLE_Compadre TRUE CACHE BOOL "Enable Compadre GMLS package")
    set(Trilinos_ENABLE_Zoltan2 TRUE CACHE BOOL "Enable Zoltan2 domain partitioning")
    set(Trilinos_ENABLE_EXPLICIT_INSTANTIATION FALSE CACHE BOOL "ETI for trilinos")

    add_subdirectory(${Trilinos_SOURCE_DIR} ${Trilinos_BINARY_DIR})

    set_property(GLOBAL PROPERTY LPM_TRILINOS_BUILT TRUE)
  endif()
endmacro()
