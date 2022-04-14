include_directories("${LPM_INCLUDE_DIRS}")

# Generate a blank lpm_version.cpp file
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/lpm_version.cpp
  COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/lpm_version.cpp
  COMMENT "Generating initial lpm_version.cpp ..."
)

# update lpm version info
add_custom_target(update_version_info ALL
  COMMAND ${BASH} ${PROJECT_SOURCE_DIR}/tools/update_version_info.sh ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/lpm_version.cpp
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/lpm_version.cpp
  COMMENT "Updating lpm version info ..."
)

add_library(lpm
                ${CMAKE_CURRENT_BINARY_DIR}/lpm_version.cpp
                lpm_bve_sphere.cpp
                lpm_bve_rk4.cpp
                lpm_comm.cpp
                lpm_coords.cpp
                lpm_error.cpp
                lpm_field.cpp
                mesh/lpm_mesh_seed.cpp
                mesh/lpm_vertices.cpp
                mesh/lpm_edges.cpp
                mesh/lpm_faces.cpp
                mesh/lpm_polymesh2d.cpp
                netcdf/lpm_netcdf.cpp
                tree/lpm_box2d.cpp
                tree/lpm_box3d.cpp
                tree/lpm_tree_node.cpp
                tree/lpm_cpu_tree.cpp
                tree/lpm_node_array_d.cpp
                tree/lpm_node_array_internal.cpp
                tree/lpm_gpu_octree.cpp
                tree/lpm_quadtree_lookup_tables.cpp
                tree/lpm_gpu_octree_lookup_tables.cpp
                tree/lpm_node_array_2d.cpp
                tree/lpm_node_array_2d_internal.cpp
                util/lpm_string_util.cpp
                util/lpm_progress_bar.cpp
                util/lpm_python3_util.cpp
                util/ekat_string_utils.cpp
                util/ekat_assert.cpp
                util/lpm_timer.cpp
                vtk/lpm_vtk_io.cpp
#    LpmCompadre.cpp
#    LpmLatLonMesh.cpp LpmMatlabIO.cpp LpmGaussGrid.cpp
#    LpmOctreeLUT.cpp
#    LpmNodeArrayD.cpp LpmNodeArrayInternal.cpp LpmOctree.cpp
#    LpmSWEGallery.cpp
                )
add_dependencies(lpm update_version_info
                     netcdf
                     ${LPM_TRILINOS_LIBS}
                     yaml-cpp
                     spdlog
                     )

target_link_libraries(lpm ${LPM_LIBRARIES} ${VTK_LIBRARIES})
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
              lpm_geometry.hpp
              lpm_logger.hpp
              lpm_sphere_functions.hpp
              mesh/lpm_mesh_seed.hpp
              mesh/lpm_vertices.hpp
              mesh/lpm_edges.hpp
              mesh/lpm_faces.hpp
              mesh/lpm_polymesh2d.hpp
              netcdf/lpm_netcdf.hpp
              netcdf/lpm_netcdf_impl.hpp
              tree/lpm_tree_node.hpp
              tree/lpm_box2d.hpp
              tree/lpm_box3d.hpp
              tree/lpm_cpu_tree.hpp
              tree/lpm_gpu_octree.hpp
              tree/lpm_gpu_octree_functions.hpp
              tree/lpm_quadtree_lookup_tables.hpp
              tree/lpm_gpu_octree_lookup_tables.hpp
              tree/lpm_tree_defs.hpp
              tree/lpm_tree_common.hpp
              tree/lpm_node_array_d.hpp
              tree/lpm_node_array_2d.hpp
              tree/lpm_node_array_internal.hpp
              tree/lpm_node_array_2d_internal.hpp
              tree/lpm_node_array_2d_internal_impl.hpp
              util/lpm_filename.hpp
              util/lpm_floating_point.hpp
              util/lpm_math.hpp
              util/lpm_progress_bar.hpp
              util/lpm_string_util.hpp
              util/lpm_timer.hpp
              util/lpm_tuple.hpp
              util/lpm_python3_util.hpp
              util/lpm_matlab_io.hpp
              util/ekat_rational_constant.hpp
              util/ekat_scaling_factor.hpp
              util/ekat_units.hpp
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
add_library(lpm_test_main lpm_catch_main.cpp)
target_link_libraries(lpm_test_main PUBLIC lpm ${CMAKE_DL_LIBS})
target_include_directories(lpm_test_main PUBLIC ${PROJECT_SOURCE_DIR}/ext/catch2/include)
install(TARGETS lpm_test_main
        EXPORT LpmTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ${PROJECT_SOURCE_DIR}/ext/catch2/include/catch.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/catch2/)

