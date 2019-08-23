INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/src" "${PROJECT_BINARY_DIR}" "${VTK_INCLUDE_DIRS}" "${Kokkos_ROOT}/include")

add_library(lpm LpmUtilities.cpp LpmCoords.cpp LpmMeshSeed.cpp LpmEdges.cpp LpmFaces.cpp LpmVtkIO.cpp 
    LpmPolyMesh2d.cpp LpmBVESphere.cpp LpmCompadre.cpp LpmLatLonMesh.cpp LpmMatlabIO.cpp)
TARGET_LINK_LIBRARIES(lpm kokkos compadre ${VTK_LIBRARIES} ${CMAKE_DL_LIBS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

install(FILES LpmConfig.h LpmDefs.hpp LpmUtilities.hpp LpmCoords.hpp LpmEdges.hpp LpmFaces.hpp LpmMeshSeed.hpp LpmVtkIO.hpp LpmCompadre.hpp LpmPolyMesh2d.hpp LpmGeometry.hpp LpmKokkosUtil.hpp LpmParticleSet.hpp LpmSpherePoisson.hpp LpmBVESphere.hpp LpmBVEKernels.hpp LpmLatLonMesh.hpp LpmRossbyWaves.hpp LpmMatlabIO.hpp
    DESTINATION include)
install(TARGETS lpm DESTINATION lib)
