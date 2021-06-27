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
    message(STATUS "Updating git submodules recursively...")
    execute_process(COMMAND ${GIT} submodule update --init --recursive
                    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    set(SUBMODULES_UPDATED TRUE)
  endif()
endmacro()


#----------------------------------------------------------------------------
# HDF5 parallel I/O library, dependency of netCDF
#----------------------------------------------------------------------------

find_package(ZLIB REQUIRED)
set(LPM_EXT_LIBRARIES ${ZLIB_LIBRARIES};${LPM_EXT_LIBRARIES})
set(LPM_EXT_INCLUDE_DIRS ${LPM_EXT_INCLUDE_DIRS};${ZLIB_INCLUDE_DIRS})

#handle user-supplied hdf5
if (HDF5_LIBRARY MATCHES ".so")
  add_library(hdf5 SHARED IMPORTED GLOBAL)
  add_library(hdf5_hl SHARED IMPORTED GLOBAL)
else()
  add_library(hdf5 STATIC IMPORTED GLOBAL)
  add_library(hdf5_hl STATIC IMPORTED GLOBAL)
endif()

set_target_properties(hdf5 PROPERTIES IMPORTED_LOCATION ${HDF5_LIBRARY})
set_target_properties(hdf5_hl PROPERTIES IMPORTED_LOCATION ${HDF5_HL_LIBRARY})

if (NOT EXISTS ${HDF5_LIBRARY})
 if (${HDF5_LIBRARY} MATCHES ${PROJECT_BINARY_DIR})
    update_submodules()

    set(HDF5_CMAKE_OPTS -DCMAKE_INSTALL_PREFIX=${PROJECT_BINARY_DIR}
                        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                        -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
                        -DCMAKE_BUILD_TYPE=RELEASE
                        -DZLIB_INCLUDE_DIRS=${ZLIB_INCLUDE_DIRS}
                        -DZLIB_LIBRARIES=${ZLIB_LIBRARIES}
                        -DZLIB_USE_EXTERNAL=OFF
                        -DHDF5_ENABLE_Z_LIB_SUPPORT=ON
                        -DHDF5_INSTALL_LIB_DIR=${CMAKE_INSTALL_LIBDIR}
                        -DHDF5_ENABLE_PARALLEL=${HAVE_MPI}
                        -DHDF5_BUILD_CPP_LIB=OFF
                        -DHDF5_BUILD_FORTRAN=OFF
                        -DHDF5_BUILD_JAVA=OFF
                        -DHDF5_BUILD_TOOLS=OFF
                        -DHDF5_BUILD_UTILS=OFF
                        -DBUILD_TESTING=OFF
                        -DHDF5_BUILD_EXAMPLES=OFF
                        )

    ExternalProject_Add(hdf5_proj
      PREFIX ${CMAKE_CURRENT_BINARY_DIR}/hdf5
      SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/hdf5
      BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/hdf5
      INSTALL_DIR ${PROJECT_BINARY_DIR}
      CMAKE_ARGS ${HDF5_CMAKE_OPTS}
      BUILD_COMMAND ${MAKE} -j
      LOG_CONFIGURE TRUE
      LOG_BUILD TRUE
      LOG_INSTALL TRUE)

    add_dependencies(hdf5 hdf5_proj)
    add_dependencies(hdf5_hl hdf5_proj)
 else()
   message(FATAL_ERROR "could not find hdf5 library; ${HDF5_LIBRARY}")
 endif()
endif()
set(LPM_EXT_INCLUDE_DIRS ${HDF5_INCLUDE_DIR};${LPM_EXT_INCLUDE_DIRS})
set(LPM_EXT_LIBRARIES hdf5_hl;hdf5;${LPM_EXT_LIBRARIES})

#----------------------------------------------------------------------------
# netCDF c
#----------------------------------------------------------------------------

add_library(netcdf STATIC IMPORTED GLOBAL)
set_target_properties(netcdf PROPERTIES IMPORTED_LOCATION ${NETCDF_LIBRARY})

if (NOT EXISTS ${NETCDF_LIBRARY})
  update_submodules()

  set(NETCDF_CMAKE_OPTS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                        -DCMAKE_BUILD_TYPE=RELEASE
                        -DCMAKE_INSTALL_PREFIX=${PROJECT_BINARY_DIR}
                        -DUSE_HDF5=ON
                        -DHDF5_C_LIBRARY=${HDF5_LIBRARY}
                        -DHDF5_HL_LIBRARY=${HDF5_HL_LIBRARY}
                        -DHDF5_C_LIBRARIES=${HDF5_LIBRARY}
                        -DHDF5_HL_LIBRARIES=${HDF5_HL_LIBRARY}
                        -DHDF5_INCLUDE_DIR=${HDF5_INCLUDE_DIR}
                        -DZLIB_LIBRARY=${ZLIB_LIBRARIES}
                        -DHAVE_LIBDL=ON
                        -DLIBDL=${CMAKE_DL_LIBS}
                        -DENABLE_PARALLEL=${HAVE_MPI}
                        -DHDF5_IS_PARALLEL=${HAVE_MPI}
                        -DHDF5_IS_PARALLEL_MPIO=${HAVE_MPI}
                        -DNC_HAVE_PARALLEL_HDF5=${HAVE_MPI}
                        -DCMAKE_INSTALL_PREFIX=${PROJECT_BINARY_DIR}
                        -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
                        -DENABLE_DAP=OFF
                        -DBUILD_SHARED_LIBS=OFF
                        -DFIND_SHARED_LIBS=OFF
                        -DENABLE_DYNAMIC_LOADING=ON
                        -DENABLE_TESTS=OFF
                        -DBUILD_UTILITIES=ON
                        -DBUILD_EXAMPLES=OFF
                        -DCMAKE_FIND_LIBRARY_SUFFIXES=.a)
  ExternalProject_Add(netcdf_proj
                      PREFIX ${CMAKE_CURRENT_BINARY_DIR}/netcdf-c
                      SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/netcdf-c
                      BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/netcdf-c
                      INSTALL_DIR ${PROJECT_BINARY_DIR}
                      CMAKE_ARGS ${NETCDF_CMAKE_OPTS}
                      DEPENDS hdf5 hdf5_hl
                      BUILD_COMMAND ${MAKE} -j
                      LOG_CONFIGURE TRUE
                      LOG_BUILD TRUE
                      BUILD_COMMAND ${MAKE}
                      LOG_INSTALL TRUE)
  add_dependencies(netcdf netcdf_proj)
endif()
list(APPEND LPM_EXT_INCLUDE_DIRS ${NETCDF_INCLUDE_DIR})
set(LPM_EXT_LIBRARIES netcdf;${LPM_EXT_LIBRARIES})

#----------------------------------------------------------------------------
# yaml-cpp
#----------------------------------------------------------------------------
add_library(yaml-cpp STATIC IMPORTED GLOBAL)
set_target_properties(yaml-cpp PROPERTIES IMPORTED_LOCATION ${YAMLCPP_LIBRARY})

if (NOT EXISTS ${YAMLCPP_LIBRARY})
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
list(APPEND LPM_EXT_INCLUDE_DIRS ${YAMLCPP_INCLUDE_DIR})
set(LPM_EXT_LIBRARIES yaml-cpp;${LPM_EXT_LIBRARIES})

#----------------------------------------------------------------------------
# Trilinos
#----------------------------------------------------------------------------

set(LPM_TRILINOS_LIBS kokkosalgorithms
                  kokkoscontainers
                  kokkoscore
                  kokkoskernels
                  teuchoscomm
                  teuchoscore
                  teuchoskokkoscomm
                  teuchoskokkoscompat
                  teuchosnumerics
                  teuchosparameterlist
                  teuchosparser
                  teuchosremainder
                  compadre
                  )

if (${HAVE_MPI})
  list(APPEND LPM_TRILINOS_LIBS tpetra
                                tpetraclassic
                                tpetraclassiclinalg
                                tpetraclassicnodeapi
                                tpetraext
                                tpetrainout
                                xpetra-sup
                                xpetra
                                zoltan
                                zoltan2
                                )
endif()

foreach(lib ${LPM_TRILINOS_LIBS})
  add_library(${lib} STATIC IMPORTED GLOBAL)
endforeach()

foreach(lib ${LPM_TRILINOS_LIBS})
  set_target_properties(${lib} PROPERTIES IMPORTED_LOCATION ${TRILINOS_LIBRARY_DIR}/lib${lib}.a)
endforeach()

if (NOT LPM_TRILINOS_NEEDS_BUILD)
foreach(lib ${LPM_TRILINOS_LIBS})
  if (NOT EXISTS "${TRILINOS_LIBRARY_DIR}/lib${lib}.a")
    message(WARNING "Trilinos library ${lib} not found in ${TRILINOS_LIBRARY_DIR}.")
  endif()
endforeach()
endif()

set(LPM_TRILINOS_LIBS ${LPM_TRILINOS_LIBS} PARENT_SCOPE)

if (LPM_TRILINOS_NEEDS_BUILD)
  update_submodules()

  set(TRILINOS_CMAKE_OPTS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                          -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                          -DCMAKE_INSTALL_PREFIX=${PROJECT_BINARY_DIR}
                          -DBUILD_SHARED_LIBS=OFF
                          -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                          -DTrilinos_ENABLE_Fortran=OFF
                          -DTPL_ENABLE_MPI=${HAVE_MPI}
                          -DTrilinos_ENABLE_OpenMP=${LPM_USE_OPENMP}
                          -DKokkos_ENABLE_OPENMP=${LPM_USE_OPENMP}
                          -DKokkos_ENABLE_SERIAL=ON
                          -DTPL_ENABLE_CUDA=${LPM_USE_CUDA}
                          -DTPL_ENABLE_CUBLAS=${LPM_USE_CUDA}
                          -DTPL_ENABLE_CUSOLVER=${LPM_USE_CUDA}
                          -DTrilinos_ENABLE_ALL_PACKAGES=OFF
                          -DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES=OFF
                          -DTrilinos_ENABLE_Teuchos=ON
                          -DTrilinos_ENABLE_Kokkos=ON
                          -DTrilinos_ENABLE_KokkosCore=ON
                          -DTrilinos_ENABLE_KokkosAlgorithms=ON
                          -DTrilinos_ENABLE_KokkosKernels=ON
                          -DKokkos_ENABLE_PROFILING=ON
                          -DKokkos_ENABLE_SERIAL=ON
                          -DKokkos_ENABLE_CUDA=${LPM_USE_CUDA}
                          -DKokkos_ARCH_${LPM_DEVICE_ARCH}=ON
                          -DTrilinos_ENABLE_Compadre=ON
                          -DTrilinos_ENABLE_Zoltan2=${HAVE_MPI}
                          -DTrilinos_ENABLE_EXPLICIT_INSTANTIATION=OFF
                          -DTPL_ENABLE_Netcdf=ON
                          -DNetcdf_INCLUDE_DIRS=${NETCDF_INCLUDE_DIR}
                          -DNetcdf_LIBRARY_DIRS=${NETCDF_LIBRARY_DIR}
                          -DTeuchos_KOKKOS_PROFILING=ON
                          )

    ExternalProject_Add(trilinos_proj
                        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/trilinos
                        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/trilinos
                        BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/trilinos
                        INSTALL_DIR ${PROJECT_BINARY_DIR}
                        CMAKE_ARGS ${TRILINOS_CMAKE_OPTS}
                        DEPENDS netcdf
                        BUILD_COMMAND ${MAKE} -j
                        LOG_CONFIGURE TRUE
                        LOG_BUILD TRUE
                        LOG_INSTALL TRUE)
    foreach(lib ${LPM_TRILINOS_LIBS})
      add_dependencies(${lib} trilinos_proj)
    endforeach()
endif()
list(APPEND LPM_EXT_INCLUDE_DIRS ${TRILINOS_INCLUDE_DIR})
foreach(lib ${LPM_TRILINOS_LIBS})
  set(LPM_EXT_LIBRARIES ${lib};${LPM_EXT_LIBRARIES})
endforeach()


#----------------------------------------------------------------------------
# VTK
#----------------------------------------------------------------------------

set(LPM_VTK_LIBS  vtkCommonColor-8.1
                  vtkCommonComputationalGeometry-8.1
                  vtkCommonCore-8.1
                  vtkCommonDataModel-8.1
                  vtkCommonExecutionModel-8.1
                  vtkCommonMath-8.1
                  vtkCommonMisc-8.1
                  vtkCommonSystem-8.1
                  vtkCommonTransforms-8.1
                  vtkDICOMParser-8.1
                  vtkDomainsChemistry-8.1
                  vtkexpat-8.1
                  vtkFiltersCore-8.1
                  vtkFiltersExtraction-8.1
                  vtkFiltersGeneral-8.1
                  vtkFiltersGeometry-8.1
                  vtkFiltersParallel-8.1
                  vtkFiltersParallelDIY2-8.1
                  vtkFiltersParallelGeometry-8.1
                  vtkFiltersParallelMPI-8.1
                  vtkFiltersParallelVerdict-8.1
                  vtkFiltersSources-8.1
                  vtkFiltersStatistics-8.1
                  vtkFiltersVerdict-8.1
                  vtkhdf5_hl-8.1
                  vtkhdf5-8.1
                  vtkImagingCore-8.1
                  vtkImagingFourier-8.1
                  vtkIOCore-8.1
                  vtkIOGeometry-8.1
                  vtkIOImage-8.1
                  vtkIOLegacy-8.1
                  vtkIOMPIImage-8.1
                  vtkIOMPIParallel-8.1
                  vtkIONetCDF-8.1
                  vtkIOParallel-8.1
                  vtkIOParallelNetCDF-8.1
                  vtkIOXML-8.1
                  vtkIOXMLParser-8.1
                  vtkjpeg-8.1
                  vtkjsoncpp-8.1
                  vtklz4-8.1
                  vtkmetaio-8.1
                  vtkNetCDF-8.1
                  vtkParallelCore-8.1
                  vtkParallelMPI-8.1
                  vtkpng-8.1
                  vtkRenderingCore-8.1
                  vtksys-8.1
                  vtktiff-8.1
                  vtkverdict-8.1
                 )

if (LPM_USE_VTK_GRAPHICS)
  set(LPM_VTK_GRAPHICS_LIBS vtkalg-8.1
                            vtkChartsCore-8.1
                            vtkDomainsChemistryOpenGL2-8.1
                            vtkexoIIc-8.1
                            vtkFiltersAMR-8.1
                            vtkFiltersFlowPaths-8.1
                            vtkFiltersGeneric-8.1
                            vtkFiltersHybrid-8.1
                            vtkFiltersHyperTree-8.1
                            vtkFiltersImaging-8.1
                            vtkFiltersModeling-8.1
                            vtkFiltersParallelImaging-8.1
                            vtkFiltersPoints-8.1
                            vtkFiltersProgrammable-8.1
                            vtkFiltersSelection-8.1
                            vtkFiltersSMP-8.1
                            vtkFiltersTexture-8.1
                            vtkFiltersTopology-8.1
                            vtkfreetype-8.1
                            vtkGeovisCore-8.1
                            vtkgl2ps-8.1
                            vtkglew-8.1
                            vtkImagingColor-8.1
                            vtkImagingGeneral-8.1
                            vtkImagingHybrid-8.1
                            vtkImagingMath-8.1
                            vtkImagingMorphological-8.1
                            vtkImagingSources-8.1
                            vtkImagingStatistics-8.1
                            vtkImagingStencil-8.1
                            vtkInfovisCore-8.1
                            vtkInfovisLayout-8.1
                            vtkInteractionImage-8.1
                            vtkInteractionStyle-8.1
                            vtkInteractionWidgets-8.1
                            vtkIOAMR-8.1
                            vtkIOEnSight-8.1
                            vtkIOExodus-8.1
                            vtkIOExport-8.1
                            vtkIOExportOpenGL2-8.1
                            vtkIOImport-8.1
                            vtkIOInfovis-8.1
                            vtkIOLSDyna-8.1
                            vtkIOMINC-8.1
                            vtkIOMovie-8.1
                            vtkIOParallelXML-8.1
                            vtkIOPLY-8.1
                            vtkIOSQL-8.1
                            vtkIOTecplotTable-8.1
                            vtkIOVideo-8.1
                            vtkharu-8.1
                            vtkxml2-8.1
                            vtknetcdfcpp-8.1
                            vtkoggtheora-8.1
                            vtkproj4-8.1
                            vtkRenderingAnnotation-8.1
                            vtkRenderingContext2D-8.1
                            vtkRenderingContextOpenGL2-8.1
                            vtkRenderingFreeType-8.1
                            vtkRenderingGL2PSOpenGL2-8.1
                            vtkRenderingImage-8.1
                            vtkRenderingLabel-8.1
                            vtkRenderingLOD-8.1
                            vtkRenderingOpenGL2-8.1
                            vtkRenderingVolume-8.1
                            vtkRenderingVolumeOpenGL2-8.1
                            vtksqlite-8.1
                            vtkViewsContext2D-8.1
                            vtkViewsCore-8.1
                            vtkViewsInfovis-8.1
                            )
    foreach(lib ${LPM_VTK_GRAPHICS_LIBS})
      list(APPEND LPM_VTK_LIBS ${lib})
    endforeach()
endif()

set(LPM_VTK_LIBS ${LPM_VTK_LIBS} PARENT_SCOPE)

foreach(lib ${LPM_VTK_LIBS})
  add_library(${lib} STATIC IMPORTED GLOBAL)
endforeach()

foreach(lib ${LPM_VTK_LIBS})
  set_target_properties(${lib} PROPERTIES IMPORTED_LOCATION "${VTK_LIBRARY_DIR}/lib${lib}.a")
endforeach()

if (NOT LPM_VTK_NEEDS_BUILD)
  foreach(lib ${LPM_VTK_LIBS})
    if (NOT EXISTS "${VTK_LIBRARY_DIR}/lib${lib}.a")
      message(WARNING "Warning: library ${lib} not found in ${VTK_LIBRARY_DIR}.")
    endif()
  endforeach()
endif()

if (LPM_VTK_NEEDS_BUILD)
  update_submodules()

  set(VTK_CMAKE_OPTS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                     -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                     -DCMAKE_INSTALL_PREFIX=${PROJECT_BINARY_DIR}
                     -DBUILD_SHARED_LIBS=OFF
                     -DCMAKE_BUILD_TYPE=RelWithDebInfo
                     -DBUILD_DOCUMENTATION=OFF
                     -DBUILD_EXAMPLES=OFF
                     -DBUILD_TESTING=OFF
                     -DVTK_Group_MPI=ON
                     -DVTK_Group_Rendering=${LPM_USE_VTK_GRAPHICS}
                     -DVTK_Group_StandAlone=ON
                     -DVTK_USE_LARGE_DATA=OFF
                     -DVTK_WRAP_JAVA=OFF
                     -DVTK_WRAP_PYTHON=OFF
                     -DVTK_PYTHON_VERSION=3
                     )

  if (NOT LPM_USE_VTK_GRAPHICS)
    list(APPEND VTK_CMAKE_OPTS -DVTK_RENDERING_BACKEND=None)
  endif()

  ExternalProject_Add(vtk_proj
                    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/vtk
                    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vtk
                    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/vtk
                    INSTALL_DIR ${PROJECT_BINARY_DIR}
                    CMAKE_ARGS ${VTK_CMAKE_OPTS}
                    BUILD_COMMAND ${MAKE} -j
                    INSTALL_COMMAND ${MAKE} install
                    LOG_CONFIGURE TRUE
                    LOG_BUILD TRUE
                    LOG_INSTALL TRUE)
  foreach(lib ${LPM_VTK_LIBS})
    add_dependencies(${lib} vtk_proj)
  endforeach()

endif()
list(APPEND LPM_EXT_INCLUDE_DIRS ${VTK_INCLUDE_DIR})
foreach(lib ${LPM_VTK_LIBS})
  set(LPM_EXT_LIBRARIES ${lib};${LPM_EXT_LIBRARIES})
endforeach()

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
if (LPM_TRILINOS_NEEDS_BUILD) # we built trilinos
  foreach(lib ${LPM_TRILINOS_LIBS})
    install(FILES lib${lib}.a DESTINATION lib)
  endforeach()
endif()
if (LPM_VTK_NEEDS_BUILD) # we built vtk
  foreach(lib ${LPM_VTK_LIBS})
    install(FILES lib${lib}.a DESTINATION lib)
  endforeach()
endif()
