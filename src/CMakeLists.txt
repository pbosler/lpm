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
                lpm_comm.cpp
                lpm_coords.cpp
                lpm_field.cpp
                lpm_bve_sphere.cpp
                mesh/lpm_mesh_seed.cpp
                mesh/lpm_vertices.cpp
                mesh/lpm_edges.cpp
                mesh/lpm_faces.cpp
                mesh/lpm_polymesh2d.cpp
                netcdf/lpm_netcdf.cpp
                vtk/lpm_vtk_io.cpp
                util/lpm_string_util.cpp
                util/lpm_progress_bar.cpp
                util/ekat_string_utils.cpp
                util/ekat_assert.cpp
                util/lpm_timer.cpp

#    LpmPolyMesh2d.cpp LpmCompadre.cpp
#    LpmLatLonMesh.cpp LpmMatlabIO.cpp LpmGaussGrid.cpp
#    LpmErrorNorms.cpp LpmBox3d.cpp  LpmOctreeLUT.cpp
#    LpmNodeArrayD.cpp LpmNodeArrayInternal.cpp LpmOctree.cpp
#    LpmSphereVoronoiPrimitives.cpp LpmSphereVoronoiMesh.cpp
#    LpmBVESphere.cpp LpmVorticityGallery.cpp LpmBVERK4.cpp
#    LpmPolyMesh2dVtkInterface.cpp LpmTimer.cpp LpmNetCDF.cpp
#    LpmSWEGallery.cpp
                )
add_dependencies(lpm update_version_info
                     netcdf
                     ${LPM_TRILINOS_LIBS}
                     ${LPM_VTK_LIBS}
                     yaml-cpp
                     spdlog
                     )

target_link_libraries(lpm ${LPM_LIBRARIES})
#if (USE_SPHEREPACK)
#    message(STATUS "linking to spherepack")
#    target_link_libraries(lpm spherepack)
#endif()

install(FILES ${PROJECT_BINARY_DIR}/LpmConfig.h
              lpm_comm.hpp
              lpm_logger.hpp
              lpm_geometry.hpp
              lpm_coords.hpp
              vtk/lpm_vtk_io.hpp
              mesh/lpm_mesh_seed.hpp
              mesh/lpm_vertices.hpp
              mesh/lpm_edges.hpp
              mesh/lpm_faces.hpp
              mesh/lpm_polymesh2d.hpp
              util/lpm_math_util.hpp
              util/lpm_floating_point_util.hpp
              util/lpm_string_util.hpp
              util/lpm_tuple.hpp
              util/lpm_timer.hpp
              util/lpm_progress_bar.hpp
              util/ekat_rational_constant.hpp
              util/ekat_scaling_factor.hpp
              util/ekat_units.hpp
              netcdf/lpm_netcdf.hpp
              netcdf/lpm_netcdf_impl.hpp

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

