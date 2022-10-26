
macro(setup_platform)



#  if (UNIX AND NOT APPLE)
#    set(LINUX ON)
#  endif()
#
 # do we have openmp?
 # TODO: Use Kokkos to determine this
 if (APPLE)
     set(LPM_USE_OPENMP FALSE)
     find_package(Threads)
 else()
  find_package(OpenMP)
  if (OpenMP_FOUND)
     message(STATUS "setting LPM_USE_OPENMP")
     set(LPM_USE_OPENMP TRUE)
  else()
    set(LPM_USE_OPENMP FALSE)
  endif()
 endif()

  if (LPM_USE_OPENMP)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    message(STATUS "OpenMP is enabled")
  endif()

  # Do we have make?
  find_program(MAKE make)
  if (MAKE STREQUAL "MAKE_NOTFOUND")
    message(FATAL_ERROR "Make is required, but is not available on this system.")
  endif()

  include(lpm_find_trilinos)
  include(lpm_build_spdlog)
  include(lpm_find_netcdf)
  include(lpm_find_vtk)


#
#  # Do we have bash?
#  find_program(BASH bash)
#  if (BASH STREQUAL "BASH_NOTFOUND")
#    message(FATAL_ERROR "Bash is required, but is not available on this system.")
#  endif()
#

#
#  # Do we have git?
#  find_program(GIT git)
#  if (GIT STREQUAL "GIT_NOTFOUND")
#    message(WARNING "Git not found. Hope you're not developing on this system.")
#    set(HAVE_GIT FALSE)
#  else()
#    set(HAVE_GIT TRUE)
#  endif()
#
#  # Do we have graphics (OpenGL)?
#  if (LPM_USE_VTK_GRAPHICS)
#    find_package(OpenGL)
#    if (NOT OPENGL_FOUND)
#      message(FATAL_ERROR "Error: vtk graphics requested, but OpenGL is not found.")
#    endif()
#  endif()
#
#  # Device type
#  if (LPM_DEVICE STREQUAL "CUDA")
#    find_package(CUDAToolkit)
#    if (NOT CUDAToolkit_FOUND)
#      message(FATAL_ERROR "Device = CUDA but CUDAToolkit is not found.")
#    endif()
#    set(LPM_USE_CUDA TRUE)
#  else()
#    set(LPM_USE_CUDA FALSE)
#  endif()
#
#  include(GNUInstallDirs)
#

#
#  #
#  # YAMLCPP (Required, will build if not found)
#  #
#  set(LPM_NEEDS_YAMLCPP_BUILD FALSE)
#  if (YAMLCPP_INCLUDE_DIR)
#    if (NOT EXISTS ${YAMLCPP_INCLUDE_DIR})
#      message(FATAL_ERROR "Couldn't find yaml-cpp include dir at ${YAMLCPP_INCLUDE_DIR}.")
#    endif()
#    message(STATUS "Using yaml-cpp include dir: ${YAMLCPP_INCLUDE_DIR}.")
#  else()
#    set(YAMLCPP_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")
#  endif()
#  if (YAMLCPP_LIBRARY)
#    if (NOT EXISTS ${YAMLCPP_LIBRARY})
#      message(FATAL_ERROR "Couldn't find yaml-cpp library at ${YAMLCPP_LIBRARY}.")
#    endif()
#    message(STATUS "Using yaml-cpp library at ${YAMLCPP_LIBRARY}.")
#  else()
#    set(YAMLCPP_LIBRARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
#    set(YAMLCPP_LIBRARY ${YAMLCPP_LIBRARY_DIR}/libyaml-cpp.a)
#    message(STATUS "Building yaml-cpp library: ${YAMLCPP_LIBRARY}.")
#    set(LPM_NEEDS_YAMLCPP_BUILD TRUE)
#  endif()

#  #
#  # Boost (optional)
#  #
#  set(LPM_USE_BOOST FALSE)
#  if (LPM_ENABLE_BOOST)
#    message(STATUS "looking for boost at $ENV{BOOST_ROOT}")
#    find_package(Boost)
#    if (Boost_FOUND)
#      set(LPM_USE_BOOST TRUE)
#    endif()
#    if (LPM_USE_BOOST)
#      message(STATUS "Using boost include dir ${Boost_INCLUDE_DIRS}.")
#    else()
#        message(FATAL_ERROR "Boost requested, but not usable.")
#    endif()
#  else()
#    message(STATUS "Boost disabled. To enable boost, set LPM_ENABLE_BOOST=TRUE")
#  endif()
#

#
#  set(LPM_LIBRARY_DIRS ${HDF5_LIBRARY_DIR};${HDF5_HL_LIBRARY_DIR};${NETCDF_LIBRARY_DIR};${YAMLCPP_LIBRARY_DIR};${SPDLOG_LIBRARY_DIR};${BOOST_LIBRARY_DIR})
#  set(LPM_INCLUDE_DIRS ${HDF5_INCLUDE_DIR};${TRILINOS_INCLUDE_DIR};${VTK_INCLUDE_DIR};${LPM_INCLUDE_DIRS};${SPDLOG_INCLUDE_DIR};${BOOST_INCLUDE_DIR})
#
#  # Certain tools (e.g. patch) require TMPDIR to be defined. If it is not,
#  # we do so here.
#  set(TMPDIR_VAR $ENV{TMPDIR})
#  if (NOT TMPDIR_VAR)
#    set(ENV{TMPDIR} "/tmp")
#  endif()
endmacro()
