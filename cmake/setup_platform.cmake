
macro(setup_platform)

  if (UNIX AND NOT APPLE)
    set(LINUX ON)
  endif()

  # do we have openmp?
  find_package(OpenMP)
  if (OpenMP-NOTFOUND)
    set(LPM_USE_OPENMP FALSE)
  else()
    set(LPM_USE_OPENMP TRUE)
  endif()
  if (APPLE)
    set(LPM_USE_OPENMP FALSE)
  endif()

  # Do we have bash?
  find_program(BASH bash)
  if (BASH STREQUAL "BASH_NOTFOUND")
    message(FATAL_ERROR "Bash is required, but is not available on this system.")
  endif()

  # Do we have make?
  find_program(MAKE make)
  if (MAKE STREQUAL "MAKE_NOTFOUND")
    message(FATAL_ERROR "Make is required, but is not available on this system.")
  endif()

  # Do we have git?
  find_program(GIT git)
  if (GIT STREQUAL "GIT_NOTFOUND")
    message(WARNING "Git not found. Hope you're not developing on this system.")
    set(HAVE_GIT FALSE)
  else()
    set(HAVE_GIT TRUE)
  endif()

  # Do we have graphics (OpenGL)?
  if (LPM_USE_VTK_GRAPHICS)
    find_package(OpenGL)
    if (NOT OPENGL_FOUND)
      message(FATAL_ERROR "Error: vtk graphics requested, but OpenGL is not found.")
    endif()
  endif()

  # Device type
  if (LPM_DEVICE STREQUAL "CUDA")
    find_package(CUDAToolkit)
    if (NOT CUDAToolkit_FOUND)
      message(FATAL_ERROR "Device = CUDA but CUDAToolkit is not found.")
    endif()
    set(LPM_USE_CUDA TRUE)
  else()
    set(LPM_USE_CUDA FALSE)
  endif()

  include(GNUInstallDirs)

  if (HDF5_INCLUDE_DIR)
    if (NOT EXISTS ${HDF5_INCLUDE_DIR})
      message(FATAL_ERROR "Couldn't find HDF5 include dir at ${HDF5_INCLUDE_DIR}.")
    endif()
    message(STATUS "Using hdf5 include dir: ${HDF5_INCLUDE_DIR}.")
  else()
    set(HDF5_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")
  endif()

  if (HDF5_LIBRARY)
    if (NOT EXISTS ${HDF5_LIBRARY})
      message(FATAL_ERROR "Couldn't find hdf5 library at ${HDF5_LIBRARY}.")
    endif()
    message(STATUS "Using hdf5 library at ${HDF5_LIBRARY}.")
  else()
    set(HDF5_LIBRARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
    set(HDF5_LIBRARY "${HDF5_LIBRARY_DIR}/libhdf5.a")
    message(STATUS "Building hdf5 library: ${HDF5_LIBRARY}.")
  endif()

  if (HDF5_HL_LIBRARY)
    if (NOT EXISTS ${HDF5_HL_LIBRARY})
      message(FATAL_ERROR "Couldn't find high-level hdf5 library at ${HDF5_HL_LIBRARY}.")
    endif()
    message(STATUS "Using high-level hdf5 library at ${HDF5_HL_LIBRARY}.")
  else()
    set(HDF5_HL_LIBRARY "${HDF5_LIBRARY_DIR}/libhdf5_hl.a")
    message(STATUS "Building high-level hdf5 library: ${HDF5_HL_LIBRARY}.")
  endif()

  get_filename_component(HDF5_LIBRARY_DIR ${HDF5_LIBRARY} DIRECTORY)
  get_filename_component(HDF5_HL_LIBRARY_DIR ${HDF5_HL_LIBRARY} DIRECTORY)

  if (NETCDF_INCLUDE_DIR)
    if (NOT EXISTS ${NETCDF_INCLUDE_DIR})
      message(FATAL_ERROR "Couldn't find NetCDF include dir at ${NETCDF_INCLUDE_DIR}.")
    endif()
    message(STATUS "Using netCDF include dir: ${NETCDF_INCLUDE_DIR}.")
  else()
    set(NETCDF_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")
  endif()
  if (NETCDF_LIBRARY)
    if (NOT EXISTS ${NETCDF_LIBRARY})
      message(FATAL_ERROR "Couldn't find NetCDF library at ${NETCDF_LIBRARY}.")
    endif()
    message(STATUS "Using netCDF library at ${NETCDF_LIBRARY}.")
  else()
    set(NETCDF_LIBRARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
    set(NETCDF_LIBRARY "${NETCDF_LIBRARY_DIR}/libnetcdf.a")
    message(STATUS "Building netCDF library: ${NETCDF_LIBRARY}.")
  endif()
  get_filename_component(NETCDF_LIBRARY_DIR ${NETCDF_LIBRARY} DIRECTORY)

  if (YAMLCPP_INCLUDE_DIR)
    if (NOT EXISTS ${YAMLCPP_INCLUDE_DIR})
      message(FATAL_ERROR "Couldn't find yaml-cpp include dir at ${YAMLCPP_INCLUDE_DIR}.")
    endif()
    message(STATUS "Using yaml-cpp include dir: ${YAMLCPP_INCLUDE_DIR}.")
  else()
    set(YAMLCPP_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")
  endif()
  if (YAMLCPP_LIBRARY)
    if (NOT EXISTS ${YAMLCPP_LIBRARY})
      message(FATAL_ERROR "Couldn't find yaml-cpp library at ${YAMLCPP_LIBRARY}.")
    endif()
    message(STATUS "Using yaml-cpp library at ${YAMLCPP_LIBRARY}.")
  else()
    set(YAMLCPP_LIBRARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
    set(YAMLCPP_LIBRARY ${YAMLCPP_LIBRARY_DIR}/libyaml-cpp.a)
    message(STATUS "Building yaml-cpp library: ${YAMLCPP_LIBRARY}.")
  endif()
  get_filename_component(YAMLCPP_LIBRARY_DIR ${YAMLCPP_LIBRARY} DIRECTORY)

  if (BOOST_INCLUDE_DIR)
    if (NOT EXISTS ${BOOST_INCLUDE_DIR})
      message(FATAL_ERROR "Couldn't find boost include dir at ${BOOST_INCLUDE_DIR}.")
    endif()
    message(STATUS "Using boost include dir ${BOOST_INCLUDE_DIR}.")
    if (NOT EXISTS ${BOOST_LIBRARY_DIR})
      message(FATAL_ERROR "Couldn't find boost library dir at ${BOOST_LIBRARY_DIR}.")
    endif()
    message(STATUS "Using boost library dir ${BOOST_LIBRARY_DIR}.")
  else()
    set(BOOST_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")
    set(BOOST_LIBRARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/lib")
    message(STATUS "Building boost at: ${BOOST_LIBRARY_DIR}.\n   Please be patient; boost can take a long time to build.")
    set(LPM_NEEDS_BOOST_BUILD TRUE)
  endif()
  set(LPM_USE_BOOST ON)

  if (TRILINOS_INCLUDE_DIR)
    if (NOT EXISTS ${TRILINOS_INCLUDE_DIR})
      message(FATAL_ERROR "Couldn't find trilinos include dir at ${TRILINOS_INCLUDE_DIR}.")
    endif()
    message(STATUS "Using trilinos include dir: ${TRILINOS_INCLUDE_DIR}.")
    if (NOT EXISTS ${TRILINOS_TPL_INCLUDE_DIR})
      message(FATAL_ERROR "Couldn't find trilinos tpl include dir at ${TRILINOS_TPL_INCLUDE_DIR}.")
    endif()
    message(STATUS "Using trilinos tpl include dir: ${TRILINOS_TPL_INCLUDE_DIR}.")
  else()
    set(TRILINOS_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")
  endif()
  if (TRILINOS_LIBRARY_DIR)
    if (NOT EXISTS ${TRILINOS_LIBRARY_DIR})
      message(FATAL_ERROR "Couldn't find trilinos library dir: ${TRILINOS_LIBRARY_DIR}.")
    endif()
    message(STATUS "Using trilinos library dir: ${TRILINOS_LIBRARY_DIR}.")
    if (NOT EXISTS ${TRILINOS_TPL_LIBRARY_DIR})
      message(FATAL_ERROR "Couldn't find trilinos tpl library dir: ${TRILINOS_TPL_LIBRARY_DIR}.")
    endif()
    message(STATUS "Using trilinos tpl library dir: ${TRILINOS_TPL_LIBRARY_DIR}.")
  else()
    set(TRILINOS_LIBRARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
    set(TRILINOS_TPL_LIBRARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
    message(STATUS "Building trilinos libraries in: ${TRILINOS_LIBRARY_DIR}.\n       Please be patient; trilinos can take a long time to build.")
    set(LPM_TRILINOS_NEEDS_BUILD TRUE)
  endif()

  find_package(VTK COMPONENTS
    vtkCommonColor
    vtkCommonCore
    vtkCommonDataModel
    vtkIOMPIParallel
    vtkIOCore
    vtkIOGeometry
    vtkIOImage
    vtkIOLegacy
    vtkIOMPIImage
    vtkIOMPIParallel
    vtkIONetCDF
    vtkIOParallel
    vtkIOParallelNetCDF
    vtkIOXML
    vtkIOXMLParser
    vtkParallelCore
    vtkParallelMPI
  )
  if (NOT VTK_FOUND)
    message("VTK not found.")
  else()
    message("Found VTK Version: ${VTK_VERSION}")
    if (VTK_VERSION VERSION_LESS "9.0.0")
      set(LPM_USE_VTK TRUE)
    else ()
      message("lpm is not compatible with this version of VTK.")
    endif()
  endif()


  if (SPDLOG_INCLUDE_DIR)
    if (NOT EXISTS ${SPDLOG_INCLUDE_DIR})
      message(FATAL_ERROR "Couldn't find spdlog include dir: ${SPDLOG_INCLUDE_DIR}")
    endif()
    message(STATUS "Using spdlog include dir: ${SPDLOG_INCLUDE_DIR}")
  else()
    set(SPDLOG_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")
  endif()
  if (SPDLOG_LIBRARY)
    if (NOT EXISTS ${SPDLOG_LIBRARY})
      message(FATAL_ERROR "Couldn't find spdlog library: ${SPDLOG_LIBRARY}")
    endif()
    message(STATUS "Using spdlog library: ${SPDLOG_LIBRARY}")
  else()
    set(SPDLOG_LIBRARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
    set(SPDLOG_LIBRARY "${SPDLOG_LIBRARY_DIR}/libspdlog.a")
    message(STATUS "Building spdlog library: ${SPDLOG_LIBRARY}")
  endif()

  set(LPM_LIBRARY_DIRS ${HDF5_LIBRARY_DIR};${HDF5_HL_LIBRARY_DIR};${NETCDF_LIBRARY_DIR};${YAMLCPP_LIBRARY_DIR};${SPDLOG_LIBRARY_DIR};${BOOST_LIBRARY_DIR})
  set(LPM_INCLUDE_DIRS ${HDF5_INCLUDE_DIR};${TRILINOS_INCLUDE_DIR};${VTK_INCLUDE_DIR};${LPM_INCLUDE_DIRS};${SPDLOG_INCLUDE_DIR};${BOOST_INCLUDE_DIR})

  # Certain tools (e.g. patch) require TMPDIR to be defined. If it is not,
  # we do so here.
  set(TMPDIR_VAR $ENV{TMPDIR})
  if (NOT TMPDIR_VAR)
    set(ENV{TMPDIR} "/tmp")
  endif()
endmacro()
