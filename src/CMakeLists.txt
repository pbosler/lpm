include_directories("${PROJECT_SOURCE_DIR}/src"
                    "${PROJECT_BINARY_DIR}"
                    "${CMAKE_CURRENT_BINARY_DIR}"
                    "${LPM_INCLUDE_DIRS}"
#                    "${VTK_INCLUDE_DIRS}"
#                   "${Trilinos_INCLUDE_DIRS}"
#                   "${Trilinos_TPL_INCLUDE_DIRS}"
#                    "${NETCDF_H_PATH}"
                    )

add_library(lpm LpmUtilities.cpp LpmCoords.cpp LpmMeshSeed.cpp LpmEdges.cpp LpmFaces.cpp
 LpmVtkIO.cpp
#    LpmPolyMesh2d.cpp LpmCompadre.cpp
#    LpmLatLonMesh.cpp LpmMatlabIO.cpp LpmGaussGrid.cpp
#    LpmErrorNorms.cpp LpmBox3d.cpp  LpmOctreeLUT.cpp
#    LpmNodeArrayD.cpp LpmNodeArrayInternal.cpp LpmOctree.cpp
#    LpmSphereVoronoiPrimitives.cpp LpmSphereVoronoiMesh.cpp
#    LpmBVESphere.cpp LpmVorticityGallery.cpp LpmBVERK4.cpp
#    LpmPolyMesh2dVtkInterface.cpp LpmTimer.cpp LpmNetCDF.cpp
#    LpmSWEGallery.cpp
)
add_dependencies(lpm update_version_info netcdf ${LPM_TRILINOS_LIBS} ${LPM_VTK_LIBS} yaml-cpp)
target_link_libraries(lpm
#      ${VTK_LIBRARIES}
      ${CMAKE_DL_LIBS}
      ${LPM_LIBRARIES}
      #${Trilinos_LIBRARIES}
      #${Trilinos_TPL_LIBRARIES}
      )
#if (USE_SPHEREPACK)
#    message(STATUS "linking to spherepack")
#    target_link_libraries(lpm spherepack)
#endif()
#if (LPM_HAS_NETCDF)
#  message(STATUS "linking to netcdf -- restart capability enabled")
#  target_link_libraries(lpm ${NETCDF_C} ${NETCDF_CXX})
#else()
#  message(STATUS "no netcdf found -- restart capability disabled")
#endif()

install(FILES LpmConfig.h LpmDefs.hpp LpmUtilities.hpp LpmCoords.hpp LpmEdges.hpp LpmFaces.hpp
              LpmMeshSeed.hpp LpmVtkIO.hpp LpmCompadre.hpp LpmPolyMesh2d.hpp LpmGeometry.hpp
              LpmKokkosUtil.hpp LpmParticleSet.hpp LpmSpherePoisson.hpp LpmBVESphere.hpp
              LpmBVEKernels.hpp LpmLatLonMesh.hpp LpmRossbyWaves.hpp LpmMatlabIO.hpp
              LpmGaussGrid.hpp LpmOctreeUtil.hpp LpmBox3d.hpp LpmNodeArrayD.hpp
              LpmNodeArrayInternal.hpp LpmOctreeLUT.hpp LpmOctree.hpp LpmVorticityGallery.hpp
              LpmBVERK4.hpp LpmBVERK4_Impl.hpp LpmPolyMesh2dVtkInterface.hpp LpmTimer.hpp
              LpmNetCDF.hpp LpmSWEGallery.hpp
    DESTINATION include)
install(TARGETS lpm DESTINATION lib)
