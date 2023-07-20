include_directories(${LPM_INCLUDE_DIRS}
                    ${Trilinos_INCLUDE_DIRS}
                    ${Trilinos_TPL_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_SOURCE_DIR}/fortran)

add_subdirectory(fortran)

if (LPM_USE_BOOST)
  include_directories(${Boost_INCLUDE_DIRS})
endif()
if (LPM_USE_NETCDF)
  include_directories(${HDF5_INCLUDE_DIRS} ${NETCDF_INCLUDE_DIRS})
endif()
if (LPM_USE_VTK)
  include_directories(${VTK_INCLUDE_DIRS})
endif()

set(LPM_SOURCES
                lpm_bve_sphere.cpp
                lpm_bve_rk4.cpp
                lpm_comm.cpp
                lpm_coords.cpp
                lpm_compadre.cpp
                lpm_error.cpp
                lpm_field.cpp
                lpm_lat_lon_pts.cpp
                lpm_dfs_grid.cpp
                mesh/lpm_mesh_seed.cpp
                mesh/lpm_vertices.cpp
                mesh/lpm_edges.cpp
                mesh/lpm_faces.cpp
                mesh/lpm_polymesh2d.cpp
                tree/lpm_box3d.cpp
                tree/lpm_cpu_tree.cpp
                util/lpm_string_util.cpp
                util/lpm_progress_bar.cpp
                util/lpm_timer.cpp)
#    LpmCompadre.cpp
#    LpmLatLonMesh.cpp LpmMatlabIO.cpp LpmGaussGrid.cpp
#    LpmOctreeLUT.cpp
#    LpmNodeArrayD.cpp LpmNodeArrayInternal.cpp LpmOctree.cpp
#    LpmSWEGallery.cpp
#                )

if (LPM_USE_NETCDF)
  LIST(APPEND LPM_SOURCES netcdf/lpm_netcdf.cpp
    netcdf/lpm_polymesh_reader.cpp
    netcdf/lpm_polymesh_writer.cpp)
endif()

if (LPM_USE_VTK)
  LIST(APPEND LPM_SOURCES vtk/lpm_vtk_io.cpp)
endif()

add_library(lpm ${LPM_SOURCES})

target_include_directories(lpm PUBLIC ${LPM_INCLUDE_DIRS})

add_dependencies(lpm spdlog)

target_link_libraries(lpm lpm_fortran
                          Kokkos::kokkos
                          Kokkos::kokkoskernels
                          Compadre::compadre
                          spdlog
                          )
if (LPM_USE_NETCDF)
  target_link_libraries(lpm ${HDF5_LIBRARY} ${HDF5_HL_LIBRARY} ${NETCDF_LIBRARY})
endif()
if (LPM_USE_COMPOSE)
  target_link_libraries(lpm compose)
endif()
if (LPM_USE_VTK)
  target_link_libraries(lpm ${VTK_LIBRARIES})
endif()

#if (USE_SPHEREPACK)
#    message(STATUS "linking to spherepack")
#    target_link_libraries(lpm spherepack)
#endif()

install(FILES ${PROJECT_BINARY_DIR}/LpmConfig.h
              lpm_bve_sphere.hpp
              lpm_bve_sphere_impl.hpp
              lpm_bve_sphere_kernels.hpp
              lpm_comm.hpp
              lpm_coords.hpp
              lpm_coords_impl.hpp
              lpm_compadre.hpp
              lpm_geometry.hpp
              lpm_logger.hpp
              lpm_sphere_functions.hpp
              lpm_lat_lon_pts.hpp
              lpm_matlab_io.hpp
              fortran/lpm_f_interp.hpp
              fortran/lpm_f_interp_impl.hpp
              mesh/lpm_mesh_seed.hpp
              mesh/lpm_vertices.hpp
              mesh/lpm_vertices_impl.hpp
              mesh/lpm_edges.hpp
              mesh/lpm_faces.hpp
              mesh/lpm_faces_impl.hpp
              mesh/lpm_polymesh2d.hpp
              mesh/lpm_polymesh2d_impl.hpp
              mesh/lpm_refinement_flags.hpp
              netcdf/lpm_netcdf.hpp
              netcdf/lpm_netcdf_impl.hpp
              tree/lpm_box3d.hpp
              tree/lpm_cpu_tree.hpp
              util/lpm_floating_point.hpp
              util/lpm_math.hpp
              util/lpm_progress_bar.hpp
              util/lpm_string_util.hpp
              util/lpm_timer.hpp
              util/lpm_tuple.hpp
              vtk/lpm_vtk_io.hpp
              vtk/lpm_vtk_io_impl.hpp


#              LpmMeshSeed.hpp LpmVtkIO.hpp LpmCompadre.hpp LpmPolyMesh2d.hpp LpmGeometry.hpp
#              LpmKokkosUtil.hpp LpmParticleSet.hpp LpmSpherePoisson.hpp LpmBVESphere.hpp
#              LpmBVEKernels.hpp LpmLatLonMesh.hpp LpmRossbyWaves.hpp LpmMatlabIO.hpp
#              LpmGaussGrid.hpp LpmOctreeUtil.hpp LpmBox3d.hpp LpmNodeArrayD.hpp
#              LpmNodeArrayInternal.hpp LpmOctreeLUT.hpp LpmOctree.hpp LpmVorticityGallery.hpp
#              LpmBVERK4.hpp LpmBVERK4_Impl.hpp LpmPolyMesh2dVtkInterface.hpp LpmTimer.hpp
#              LpmNetCDF.hpp LpmSWEGallery.hpp
    DESTINATION include)
install(TARGETS lpm DESTINATION lib)

###########################################
###       Testing micro-libraries       ###
###########################################

# Pre-compile lpm_catch_main.cpp into its own mini-library, so that individual
# unit test don't have to all rebuild the same file.
add_library(lpm_test_main util/lpm_catch_main.cpp)
add_dependencies(lpm_test_main catch2)
target_link_libraries(lpm_test_main PUBLIC lpm catch2 ${CMAKE_DL_LIBS})
target_include_directories(lpm_test_main PUBLIC ${CATCH2_INCLUDE_DIR})
install(TARGETS lpm_test_main
        EXPORT LpmTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

