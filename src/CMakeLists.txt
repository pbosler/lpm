include_directories("${LPM_INCLUDE_DIRS}")

add_library(lpm
#                LpmUtilities.cpp
#                LpmCoords.cpp
#                LpmMeshSeed.cpp
#                LpmEdges.cpp
#                LpmFaces.cpp
#                LpmVtkIO.cpp
                lpm_comm.cpp
                lpm_mesh_seed.cpp
                lpm_coords.cpp
                util/lpm_string_util.cpp
                util/lpm_progress_bar.cpp

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

install(FILES LpmConfig.h
              lpm_comm.hpp
              lpm_logger.hpp
              lpm_geometry.hpp
              lpm_mesh_seed.hpp
              lpm_coords.hpp
              util/lpm_math_util.hpp
              util/lpm_floating_point_util.hpp
              util/lpm_string_util.hpp
              util/lpm_tuple.hpp
              util/lpm_timer.hpp
              util/lpm_progress_bar.hpp
              LpmCoords.hpp LpmEdges.hpp LpmFaces.hpp
              LpmMeshSeed.hpp LpmVtkIO.hpp LpmCompadre.hpp LpmPolyMesh2d.hpp LpmGeometry.hpp
              LpmKokkosUtil.hpp LpmParticleSet.hpp LpmSpherePoisson.hpp LpmBVESphere.hpp
              LpmBVEKernels.hpp LpmLatLonMesh.hpp LpmRossbyWaves.hpp LpmMatlabIO.hpp
              LpmGaussGrid.hpp LpmOctreeUtil.hpp LpmBox3d.hpp LpmNodeArrayD.hpp
              LpmNodeArrayInternal.hpp LpmOctreeLUT.hpp LpmOctree.hpp LpmVorticityGallery.hpp
              LpmBVERK4.hpp LpmBVERK4_Impl.hpp LpmPolyMesh2dVtkInterface.hpp LpmTimer.hpp
              LpmNetCDF.hpp LpmSWEGallery.hpp
    DESTINATION include)
install(TARGETS lpm DESTINATION lib)

###########################################
###       Testing micro-libraries       ###
###########################################

# Pre-compile lpm_catch_main.cpp into its own mini-library, so that individual
# unit test don't have to all rebuild the same file.
add_library(lpm_test_main lpm_catch_main.cpp)
target_link_libraries(lpm_test_main PUBLIC lpm)
target_include_directories(lpm_test_main PUBLIC ${PROJECT_SOURCE_DIR}/ext/catch2/include)
install(TARGETS lpm_test_main
        EXPORT LpmTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY ${PROJECT_SOURCE_DIR}/ext/catch2/include
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

