INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/src" "${PROJECT_BINARY_DIR}"
                    "${VTK_INCLUDE_DIRS}"
                    "${Trilinos_INCLUDE_DIRS}"
                    "${Trilinos_TPL_INCLUDE_DIRS}")

add_library(lpm LpmUtilities.cpp LpmCoords.cpp LpmMeshSeed.cpp LpmEdges.cpp LpmFaces.cpp LpmVtkIO.cpp
    LpmPolyMesh2d.cpp LpmCompadre.cpp
    LpmLatLonMesh.cpp LpmMatlabIO.cpp LpmGaussGrid.cpp
    LpmErrorNorms.cpp LpmBox3d.cpp  LpmOctreeLUT.cpp
    LpmNodeArrayD.cpp LpmNodeArrayInternal.cpp LpmOctree.cpp
    LpmSphereVoronoiPrimitives.cpp LpmSphereVoronoiMesh.cpp
    LpmBVESphere.cpp LpmVorticityGallery.cpp LpmRK4.cpp
    LpmPolyMesh2dVtkInterface.cpp
)
TARGET_LINK_LIBRARIES(lpm ${Trilinos_LIBRARIES} ${VTK_LIBRARIES} ${CMAKE_DL_LIBS} ${Trilinos_TPL_LIBRARIES})
if (USE_SPHEREPACK)
    message(STATUS "linking to spherepack")
    TARGET_LINK_LIBRARIES(lpm spherepack)
endif()

install(FILES LpmConfig.h LpmDefs.hpp LpmUtilities.hpp LpmCoords.hpp LpmEdges.hpp LpmFaces.hpp
              LpmMeshSeed.hpp LpmVtkIO.hpp LpmCompadre.hpp LpmPolyMesh2d.hpp LpmGeometry.hpp
              LpmKokkosUtil.hpp LpmParticleSet.hpp LpmSpherePoisson.hpp LpmBVESphere.hpp
              LpmBVEKernels.hpp LpmLatLonMesh.hpp LpmRossbyWaves.hpp LpmMatlabIO.hpp
              LpmGaussGrid.hpp LpmOctreeUtil.hpp LpmBox3d.hpp LpmNodeArrayD.hpp
              LpmNodeArrayInternal.hpp LpmOctreeLUT.hpp LpmOctree.hpp LpmVorticityGallery.hpp
              LpmRK4.hpp LpmRK4_Impl.hpp LpmPolyMesh2dVtkInterface.hpp
    DESTINATION include)
install(TARGETS lpm DESTINATION lib)
