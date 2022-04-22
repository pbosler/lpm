include(ExternalProject)
include(GNUInstallDirs)
include(${PROJECT_SOURCE_DIR}/cmake/cmake_helpers.cmake)
include_directories(${PROJECT_BINARY_DIR})

# create top-level directories for each third-party library (tpl)
foreach(dir include;lib;bin;share)
  set(top_level_dir ${PROJECT_BINARY_DIR}/${dir})
  if (NOT EXISTS ${top_level_dir})
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${top_level_dir})
  endif()
endforeach()

# keep track of submodules
set(SUBMODULES_UPDATED FALSE)
macro(update_submodules)
  if (NOT SUBMODULES_UPDATED)
#    message(STATUS "Updating git submodules recursively...")
#    execute_process(COMMAND ${GIT} submodule update --init --recursive
#                    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    message(WARNING "git submodules may need updating...\n...consider running 'git submodule update --init --recursive'")
    set(SUBMODULES_UPDATED TRUE)
  endif()
endmacro()

#----------------------------------------------------------------------------
# Catch2 test framework
#----------------------------------------------------------------------------
list(APPEND LPM_EXT_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/ext/catch2/include)

#----------------------------------------------------------------------------
# spdlog logging utilities
#----------------------------------------------------------------------------

add_library(spdlog STATIC IMPORTED GLOBAL)
set_target_properties(spdlog PROPERTIES IMPORTED_LOCATION ${SPDLOG_LIBRARY})
list(APPEND LPM_EXT_INCLUDE_DIRS ${SPDLOG_INCLUDE_DIR})
set(LPM_EXT_LIBRARIES spdlog;${LPM_EXT_LIBRARIES})
if (LPM_NEEDS_SPDLOG_BUILD)
  update_submodules()
  set(SPDLOG_CMAKE_OPTS -DCMAKE_INSTALL_PREFIX=${PROJECT_BINARY_DIR}
                      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                      -DCMAKE_BUILD_TYPE=RelWithDebInfo
                      -DBUILD_SHARED_LIBS=OFF
                      -DSPDLOG_BUILD_EXAMPLES=OFF
                      -DSPDLOG_BUILD_TESTS=OFF
                      -DSPDLOG_INSTALL=ON
                      )

  ExternalProject_Add(spdlog_proj
          PREFIX ${CMAKE_CURRENT_BINARY_DIR}/spdlog
          SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/spdlog
          BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/spdlog
          INSTALL_DIR ${PROJECT_BINARY_DIR}
          CMAKE_ARGS ${SPDLOG_CMAKE_OPTS}
          BUILD_COMMAND ${MAKE} -j
          LOG_CONFIGURE TRUE
          LOG_BUILD TRUE
          LOG_INSTALL TRUE
          )
  add_dependencies(spdlog spdlog_proj)
endif()



#----------------------------------------------------------------------------
# yaml-cpp
#----------------------------------------------------------------------------
list(APPEND LPM_EXT_INCLUDE_DIRS ${YAMLCPP_INCLUDE_DIR})
set(LPM_EXT_LIBRARIES yaml-cpp;${LPM_EXT_LIBRARIES})
add_library(yaml-cpp STATIC IMPORTED GLOBAL)
set_target_properties(yaml-cpp PROPERTIES IMPORTED_LOCATION ${YAMLCPP_LIBRARY})
if (LPM_NEEDS_YAMLCPP_BUILD)
  update_submodules()

  set(YAMLCPP_CMAKE_OPTS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                          -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                          -DCMAKE_INSTALL_PREFIX=${PROJECT_BINARY_DIR}
                          -DYAML_CPP_BUILD_SHARED_LIBS=OFF
                          -DYAML_CPP_BUILD_TESTS=OFF
                          -DYAML_CPP_BUILD_TOOLS=OFF
                          )

  ExternalProject_Add(yaml-cpp_proj
                      PREFIX ${CMAKE_CURRENT_BINARY_DIR}/yaml-cpp
                      SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/yaml-cpp
                      BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/yaml-cpp
                      INSTALL_DIR ${PROJECT_BINARY_DIR}
                      CMAKE_ARGS ${YAMLCPP_CMAKE_OPTS}
                      BUILD_COMMAND ${MAKE} -j
                      LOG_CONFIGURE TRUE
                      LOG_BUILD TRUE
                      LOG_INSTALL TRUE
                      )

  add_dependencies(yaml-cpp yaml-cpp_proj)
endif()
#get_filename_component(YAMLCPP_LIBRARY_DIR ${YAMLCPP_LIBRARY} DIRECTORY)


#----------------------------------------------------------------------------
# HDF5 parallel I/O library, dependency of netCDF
#----------------------------------------------------------------------------
if (LPM_USE_NETCDF)
  set(LPM_EXT_LIBRARIES ${ZLIB_LIBRARIES};${LPM_EXT_LIBRARIES})
  list(APPEND LPM_EXT_INCLUDE_DIRS ${ZLIB_INCLUDE_DIRS})

  if (HDF5_LIBRARY MATCHES ".so")
    add_library(hdf5 SHARED IMPORTED GLOBAL)
    add_library(hdf5_hl SHARED IMPORTED GLOBAL)
  else()
    add_library(hdf5 STATIC IMPORTED GLOBAL)
    add_library(hdf5_hl STATIC IMPORTED GLOBAL)
  endif()
  set_target_properties(hdf5 PROPERTIES IMPORTED_LOCATION ${HDF5_LIBRARY})
  set_target_properties(hdf5_hl PROPERTIES IMPORTED_LOCATION ${HDF5_HL_LIBRARY})

#if (NOT EXISTS ${HDF5_LIBRARY})
# if (${HDF5_LIBRARY} MATCHES ${PROJECT_BINARY_DIR})
#    update_submodules()
#
#    set(HDF5_CMAKE_OPTS -DCMAKE_INSTALL_PREFIX=${PROJECT_BINARY_DIR}
#                        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
#                        -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
#                        -DCMAKE_BUILD_TYPE=RELEASE
#                        -DZLIB_INCLUDE_DIRS=${ZLIB_INCLUDE_DIRS}
#                        -DZLIB_LIBRARIES=${ZLIB_LIBRARIES}
#                        -DZLIB_USE_EXTERNAL=OFF
#                        -DHDF5_ENABLE_Z_LIB_SUPPORT=ON
#                        -DHDF5_INSTALL_LIB_DIR=${CMAKE_INSTALL_LIBDIR}
#                        -DHDF5_ENABLE_PARALLEL=${HAVE_MPI}
#                        -DHDF5_BUILD_CPP_LIB=OFF
#                        -DHDF5_BUILD_FORTRAN=OFF
#                        -DHDF5_BUILD_JAVA=OFF
#                        -DHDF5_BUILD_TOOLS=OFF
#                        -DHDF5_BUILD_UTILS=OFF
#                        -DBUILD_TESTING=OFF
#                        -DHDF5_BUILD_EXAMPLES=OFF
#                        )
#
#    ExternalProject_Add(hdf5_proj
#      PREFIX ${CMAKE_CURRENT_BINARY_DIR}/hdf5
#      SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/hdf5
#      BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/hdf5
#      INSTALL_DIR ${PROJECT_BINARY_DIR}
#      CMAKE_ARGS ${HDF5_CMAKE_OPTS}
#      BUILD_COMMAND ${MAKE} -j
#      LOG_CONFIGURE TRUE
#      LOG_BUILD TRUE
#      LOG_INSTALL TRUE)
#
#    add_dependencies(hdf5 hdf5_proj)
#    add_dependencies(hdf5_hl hdf5_proj)
# else()
#   message(FATAL_ERROR "could not find hdf5 library; ${HDF5_LIBRARY}")
# endif()
#endif()
#list(APPEND LPM_EXT_INCLUDE_DIRS ${HDF5_INCLUDE_DIR})
#set(LPM_EXT_LIBRARIES hdf5_hl;hdf5;${LPM_EXT_LIBRARIES})

    #----------------------------------------------------------------------------
    # netCDF c
    #----------------------------------------------------------------------------
    list(APPEND LPM_EXT_INCLUDE_DIRS ${NETCDF_INCLUDE_DIR})
    set(LPM_EXT_LIBRARIES netcdf;${LPM_EXT_LIBRARIES})
    add_library(netcdf STATIC IMPORTED GLOBAL)
    set_target_properties(netcdf PROPERTIES IMPORTED_LOCATION ${NETCDF_LIBRARY})
#if (NOT EXISTS ${NETCDF_LIBRARY})
#  update_submodules()
#
#  set(NETCDF_CMAKE_OPTS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
#                        -DCMAKE_BUILD_TYPE=RELEASE
#                        -DCMAKE_INSTALL_PREFIX=${PROJECT_BINARY_DIR}
#                        -DUSE_HDF5=ON
#                        -DHDF5_C_LIBRARY=${HDF5_LIBRARY}
#                        -DHDF5_HL_LIBRARY=${HDF5_HL_LIBRARY}
#                        -DHDF5_C_LIBRARIES=${HDF5_LIBRARY}
#                        -DHDF5_HL_LIBRARIES=${HDF5_HL_LIBRARY}
#                        -DHDF5_INCLUDE_DIR=${HDF5_INCLUDE_DIR}
#                        -DZLIB_LIBRARY=${ZLIB_LIBRARIES}
#                        -DHAVE_LIBDL=ON
#                        -DLIBDL=${CMAKE_DL_LIBS}
#                        -DENABLE_PARALLEL=${HAVE_MPI}
#                        -DHDF5_IS_PARALLEL=${HAVE_MPI}
#                        -DHDF5_IS_PARALLEL_MPIO=${HAVE_MPI}
#                        -DNC_HAVE_PARALLEL_HDF5=${HAVE_MPI}
#                        -DCMAKE_INSTALL_PREFIX=${PROJECT_BINARY_DIR}
#                        -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
#                        -DENABLE_DAP=OFF
#                        -DBUILD_SHARED_LIBS=OFF
#                        -DFIND_SHARED_LIBS=OFF
#                        -DENABLE_DYNAMIC_LOADING=ON
#                        -DENABLE_TESTS=OFF
#                        -DBUILD_UTILITIES=ON
#                        -DBUILD_EXAMPLES=OFF
#                        -DCMAKE_FIND_LIBRARY_SUFFIXES=.a)
#  ExternalProject_Add(netcdf_proj
#                      PREFIX ${CMAKE_CURRENT_BINARY_DIR}/netcdf-c
#                      SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/netcdf-c
#                      BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/netcdf-c
#                      INSTALL_DIR ${PROJECT_BINARY_DIR}
#                      CMAKE_ARGS ${NETCDF_CMAKE_OPTS}
#                      DEPENDS hdf5 hdf5_hl
#                      BUILD_COMMAND ${MAKE} -j
#                      LOG_CONFIGURE TRUE
#                      LOG_BUILD TRUE
#                      BUILD_COMMAND ${MAKE}
#                      LOG_INSTALL TRUE)
#  add_dependencies(netcdf netcdf_proj)
#endif()
endif()



#----------------------------------------------------------------------------
# Boost
#----------------------------------------------------------------------------
#if (LPM_ENABLE_BOOST)
#set(LPM_BOOST_LIBS boost_math_c99
#                   boost_math_c99f
#                   boost_math_c99l
#                   boost_math_tr1
#                   boost_math_tr1f
#                   boost_math_tr1l
#                   )
#foreach(lib ${LPM_BOOST_LIBS})
#  add_library(${lib} STATIC IMPORTED GLOBAL)
#  set_target_properties(${lib} PROPERTIES IMPORTED_LOCATION ${BOOST_LIBRARY_DIR}/lib${lib}.a)
#endforeach()
#
##if (NOT LPM_NEEDS_BOOST_BUILD)
##  foreach(lib ${LPM_BOOST_LIBS})
##    if (NOT EXISTS "${BOOST_LIBRARY_DIR}/lib${lib}.a")
##      message(WARNING "Boost library ${lib} not found in ${BOOST_LIBRARY_DIR}.")
##    endif()
##  endforeach()
##endif()
#
#set(LPM_BOOST_LIBS ${LPM_BOOST_LIBS} PARENT_SCOPE)
#
#if (LPM_NEEDS_BOOST_BUILD)
#  update_submodules()
#
#  ExternalProject_Add(boost_proj
#      PREFIX ${CMAKE_CURRENT_BINARY_DIR}/boost
#      SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/boost
#      BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/boost
#      INSTALL_DIR ${PROJECT_BINARY_DIR}
#      CONFIGURE_COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/boost/bootstrap.sh --with-libraries=math --prefix=${PROJECT_BINARY_DIR}"
#      BUILD_COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/boost/b2 -j"
#      INSTALL_COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/boost/b2 install"
#      LOG_CONFIGURE TRUE
#      LOG_BUILD TRUE
#      LOG_INSTALL TRUE)
#endif()
#endif()

#----------------------------------------------------------------------------
# Trilinos
#----------------------------------------------------------------------------

#set(LPM_TRILINOS_LIBS kokkosalgorithms
#                  kokkoscontainers
#                  kokkoscore
#                  kokkoskernels
#                  teuchoscomm
#                  teuchoscore
#                  teuchoskokkoscomm
#                  teuchoskokkoscompat
#                  teuchosnumerics
#                  teuchosparameterlist
#                  teuchosparser
#                  teuchosremainder
#                  compadre
#                  )
#
#if (${HAVE_MPI})
#  list(APPEND LPM_TRILINOS_LIBS tpetra
#                                tpetraclassic
#                                tpetraclassiclinalg
#                                tpetraclassicnodeapi
#                                tpetraext
#                                tpetrainout
#                                xpetra-sup
#                                xpetra
#                                zoltan
#                                zoltan2
#                                )
#endif()
#
#foreach(lib ${LPM_TRILINOS_LIBS})
#  add_library(${lib} STATIC IMPORTED GLOBAL)
#endforeach()
#
#foreach(lib ${LPM_TRILINOS_LIBS})
#  set_target_properties(${lib} PROPERTIES IMPORTED_LOCATION ${TRILINOS_LIBRARY_DIR}/lib${lib}.a)
#endforeach()
#
#if (NOT LPM_TRILINOS_NEEDS_BUILD)
#foreach(lib ${LPM_TRILINOS_LIBS})
#  if (NOT EXISTS "${TRILINOS_LIBRARY_DIR}/lib${lib}.a")
#    message(WARNING "Trilinos library ${lib} not found in ${TRILINOS_LIBRARY_DIR}.")
#  endif()
#endforeach()
#endif()
#
#set(LPM_TRILINOS_LIBS ${LPM_TRILINOS_LIBS} PARENT_SCOPE)
#
#if (LPM_TRILINOS_NEEDS_BUILD)
#  update_submodules()
#
#  set(TRILINOS_CMAKE_OPTS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
#                          -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
#                          -DCMAKE_INSTALL_PREFIX=${PROJECT_BINARY_DIR}
#                          -DBUILD_SHARED_LIBS=OFF
#                          -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
#                          -DTrilinos_ENABLE_Fortran=OFF
#                          -DTPL_ENABLE_MPI=${HAVE_MPI}
#                          -DTrilinos_ENABLE_OpenMP=${LPM_USE_OPENMP}
#                          -DKokkos_ENABLE_OPENMP=${LPM_USE_OPENMP}
#                          -DKokkos_ENABLE_SERIAL=ON
#                          -DTPL_ENABLE_CUDA=${LPM_USE_CUDA}
#                          -DTPL_ENABLE_CUBLAS=${LPM_USE_CUDA}
#                          -DTPL_ENABLE_CUSOLVER=${LPM_USE_CUDA}
#                          -DTrilinos_ENABLE_ALL_PACKAGES=OFF
#                          -DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES=OFF
#                          -DTrilinos_ENABLE_Teuchos=ON
#                          -DTrilinos_ENABLE_Kokkos=ON
#                          -DTrilinos_ENABLE_KokkosCore=ON
#                          -DTrilinos_ENABLE_KokkosAlgorithms=ON
#                          -DTrilinos_ENABLE_KokkosKernels=ON
#                          -DKokkos_ENABLE_PROFILING=ON
#                          -DKokkos_ENABLE_SERIAL=ON
#                          -DKokkos_ENABLE_CUDA=${LPM_USE_CUDA}
#                          -DKokkos_ARCH_${LPM_DEVICE_ARCH}=ON
#                          -DTrilinos_ENABLE_Compadre=ON
#                          -DTrilinos_ENABLE_Zoltan2=${HAVE_MPI}
#                          -DTrilinos_ENABLE_EXPLICIT_INSTANTIATION=OFF
#                          -DTPL_ENABLE_Netcdf=ON
#                          -DNetcdf_INCLUDE_DIRS=${NETCDF_INCLUDE_DIR}
#                          -DNetcdf_LIBRARY_DIRS=${NETCDF_LIBRARY_DIR}
#                          -DTeuchos_KOKKOS_PROFILING=ON
#                          )
#
#    ExternalProject_Add(trilinos_proj
#                        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/trilinos
#                        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/trilinos
#                        BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/trilinos
#                        INSTALL_DIR ${PROJECT_BINARY_DIR}
#                        CMAKE_ARGS ${TRILINOS_CMAKE_OPTS}
#                        DEPENDS netcdf
#                        BUILD_COMMAND ${MAKE} -j
#                        LOG_CONFIGURE TRUE
#                        LOG_BUILD TRUE
#                        LOG_INSTALL TRUE)
#    foreach(lib ${LPM_TRILINOS_LIBS})
#      add_dependencies(${lib} trilinos_proj)
#    endforeach()
#endif()
#list(APPEND LPM_EXT_INCLUDE_DIRS ${TRILINOS_INCLUDE_DIR})
#foreach(lib ${LPM_TRILINOS_LIBS})
#  set(LPM_EXT_LIBRARIES ${lib};${LPM_EXT_LIBRARIES})
#endforeach()


#----------------------------------------------------------------------------
# Add all external libraries to the build system
#----------------------------------------------------------------------------
set(LPM_EXT_LIBRARIES ${CMAKE_DL_LIBS};${LPM_EXT_LIBRARIES})
set(LPM_EXT_LIBRARIES ${LPM_EXT_LIBRARIES} PARENT_SCOPE)
list(REMOVE_DUPLICATES LPM_EXT_INCLUDE_DIRS)
set(LPM_EXT_INCLUDE_DIRS ${LPM_EXT_INCLUDE_DIRS} PARENT_SCOPE)
set(LPM_LIBRARIES ${LPM_EXT_LIBRARIES};${LPM_LIBRARIES} PARENT_SCOPE)
set(LPM_INCLUDE_DIRS ${LPM_INCLUDE_DIRS};${LPM_EXT_INCLUDE_DIRS} PARENT_SCOPE)

#----------------------------------------------------------------------------
# Installation targets
#----------------------------------------------------------------------------

install(DIRECTORY ${PROJECT_BINARY_DIR}/include/ DESTINATION include)
if (${HDF5_LIBRARY_DIR} MATCHES ${PROJECT_BINARY_DIR}) # we built hdf5
  install(FILES ${HDF5_LIBRARY} ${HDF5_HL_LIBRARY} DESTINATION lib)
endif()
if (${NETCDF_LIBRARY_DIR} MATCHES ${PROJECT_BINARY_DIR}) # we built netcdf
  install(FILES ${NETCDF_LIBRARY} DESTINATION lib)
endif()
if (${YAMLCPP_LIBRARY_DIR} MATCHES ${PROJECT_BINARY_DIR}) # we built yamlcpp
  install(FILES ${YAMLCPP_LIBRARY} DESTINATION lib)
endif()
#if (LPM_TRILINOS_NEEDS_BUILD) # we built trilinos
#  foreach(lib ${LPM_TRILINOS_LIBS})
#    install(FILES lib${lib}.a DESTINATION lib)
#  endforeach()
#endif()
#if (LPM_VTK_NEEDS_BUILD) # we built vtk
#  foreach(lib ${LPM_VTK_LIBS})
#    install(FILES lib${lib}.a DESTINATION lib)
#  endforeach()
#endif()
