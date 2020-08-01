include (CTest)
INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/src"
                    "${PROJECT_BINARY_DIR}"
                    "${VTK_INCLUDE_DIRS}"
                    "${Trilinos_INCLUDE_DIRS}"
                    "${Trilinos_TPL_INCLUDE_DIRS}"
)

ADD_EXECUTABLE(lpmGeomTest LpmGeometryUnitTests.cpp)
TARGET_LINK_LIBRARIES(lpmGeomTest lpm ${Trilinos_LIBRARIES} ${VTK_LIBRARIES} ${Trilinos_TPL_LIBRARIES})
ADD_TEST(lpmGeom lpmGeomTest)

ADD_EXECUTABLE(lpmCoordsTest LpmCoordsUnitTest.cpp)
TARGET_LINK_LIBRARIES(lpmCoordsTest lpm ${Trilinos_LIBRARIES} ${VTK_LIBRARIES} ${Trilinos_TPL_LIBRARIES})
ADD_TEST(lpmCoords lpmCoordsTest)

ADD_EXECUTABLE(lpmEdgesTest LpmEdgesUnitTest.cpp)
TARGET_LINK_LIBRARIES(lpmEdgesTest lpm ${Trilinos_LIBRARIES} ${VTK_LIBRARIES} ${Trilinos_TPL_LIBRARIES})
ADD_TEST(lpmEdges lpmEdgesTest)

ADD_EXECUTABLE(lpmFacesTest LpmFacesUnitTests.cpp)
TARGET_LINK_LIBRARIES(lpmFacesTest lpm ${Trilinos_LIBRARIES} ${VTK_LIBRARIES} ${Trilinos_TPL_LIBRARIES})
ADD_TEST(lpmFaces lpmFacesTest)

ADD_EXECUTABLE(lpmMeshSeedTest LpmMeshSeedUnitTest.cpp)
TARGET_LINK_LIBRARIES(lpmMeshSeedTest lpm ${Trilinos_LIBRARIES} ${VTK_LIBRARIES} ${Trilinos_TPL_LIBRARIES})
ADD_TEST(lpmMeshSeed lpmMeshSeedTest)

ADD_EXECUTABLE(lpmPolyMeshTest LpmPolyMeshTest.cpp)
TARGET_LINK_LIBRARIES(lpmPolyMeshTest lpm ${Trilinos_LIBRARIES} ${VTK_LIBRARIES} ${Trilinos_TPL_LIBRARIES})
ADD_TEST(lpmPolyMeshTest lpmPolyMeshTest)

ADD_EXECUTABLE(lpmKernelTest LpmKernelTest.cpp)
TARGET_LINK_LIBRARIES(lpmKernelTest lpm ${Trilinos_LIBRARIES} ${VTK_LIBRARIES} ${Trilinos_TPL_LIBRARIES})
ADD_TEST(lpmKernelTest lpmKernelTest)

ADD_EXECUTABLE(lpmSpherePoissonTest LpmSpherePoisson.cpp)
TARGET_LINK_LIBRARIES(lpmSpherePoissonTest lpm ${Trilinos_LIBRARIES} ${VTK_LIBRARIES} ${Trilinos_TPL_LIBRARIES})
ADD_TEST(lpmSpherePoissonTest lpmSpherePoissonTest)

#ADD_EXECUTABLE(lpmGMLS LpmGMLSTest.cpp)
##SET_TARGET_PROPERTIES(lpmGMLS PROPERTIES COMPILE_FLAGS "${LPM_CXXFLAGS}" LINK_FLAGS "${LPM_LDFLAGS}")
#TARGET_LINK_LIBRARIES(lpmGMLS lpm compadre ${Trilinos_LIBRARIES} ${VTK_LIBRARIES} ${CMAKE_DL_LIBS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} )
#ADD_TEST(lpmGMLS lpmGMLS)

ADD_EXECUTABLE(lpmLatLonMesh LpmLatLonMeshTests.cpp)
TARGET_LINK_LIBRARIES(lpmLatLonMesh lpm ${Trilinos_LIBRARIES} ${VTK_LIBRARIES} ${CMAKE_DL_LIBS}  ${Trilinos_TPL_LIBRARIES})
ADD_TEST(lpmLatLonMeshTest lpmLatLonMesh)

ADD_EXECUTABLE(lpmCompadre LpmCompadreTests.cpp)
TARGET_LINK_LIBRARIES(lpmCompadre lpm ${Trilinos_LIBRARIES} ${VTK_LIBRARIES} ${CMAKE_DL_LIBS} ${Trilinos_TPL_LIBRARIES})
ADD_TEST(lpmCompadreTest lpmCompadre)

#ADD_EXECUTABLE(lpmGaussGrid LpmGaussGridTests.cpp)
#TARGET_LINK_LIBRARIES(lpmGaussGrid lpm compadre ${Trilinos_LIBRARIES} ${VTK_LIBRARIES} ${CMAKE_DL_LIBS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
#ADD_TEST(lpmGaussGridTest lpmGaussGrid)

ADD_EXECUTABLE(lpmOctreeKeyTests LpmOctreeKeyTests.cpp)
TARGET_LINK_LIBRARIES(lpmOctreeKeyTests lpm ${Trilinos_LIBRARIES} ${CMAKE_DL_LIBS} ${Trilinos_TPL_LIBRARIES})
ADD_TEST(lpmOctreeKeyTests lpmOctreeKeyTests)

ADD_EXECUTABLE(lpmOctreeLUTUnitTest LpmOctreeLUTUnitTests.cpp)
TARGET_LINK_LIBRARIES(lpmOctreeLUTUnitTest lpm ${Trilinos_LIBRARIES} ${CMAKE_DL_LIBS} ${Trilinos_TPL_LIBRARIES})
ADD_TEST(lpmOctreeLUTUnitTest lpmOctreeLUTUnitTest)

ADD_EXECUTABLE(lpmOctreeUtilsTest LpmOctreeUtilsUnitTests.cpp)
TARGET_LINK_LIBRARIES(lpmOctreeUtilsTest lpm ${Trilinos_LIBRARIES} ${CMAKE_DL_LIBS} ${Trilinos_TPL_LIBRARIES})
ADD_TEST(lpmOctreeUtilsTest lpmOctreeUtilsTest)

ADD_EXECUTABLE(lpmOctreeKernelTests LpmOctreeKernelTests.cpp)
TARGET_LINK_LIBRARIES(lpmOctreeKernelTests lpm ${Trilinos_LIBRARIES} ${CMAKE_DL_LIBS} ${Trilinos_TPL_LIBRARIES})
ADD_TEST(lpmOctreeKernelTests lpmOctreeKernelTests)

ADD_EXECUTABLE(lpmNodeArrayDTest LpmNodeArrayDUnitTests.cpp)
TARGET_LINK_LIBRARIES(lpmNodeArrayDTest lpm ${Trilinos_LIBRARIES} ${CMAKE_DL_LIBS} ${Trilinos_TPL_LIBRARIES})
ADD_TEST(lpmNodeArrayDTest lpmNodeArrayDTest)

ADD_EXECUTABLE(lpmNodeArrayInternalTest LpmNodeArrayInternalTests.cpp)
TARGET_LINK_LIBRARIES(lpmNodeArrayInternalTest lpm ${Trilinos_LIBRARIES} ${CMAKE_DL_LIBS} ${Trilinos_TPL_LIBRARIES})
ADD_TEST(lpmNodeArrayInternalTest lpmNodeArrayInternalTest)

ADD_EXECUTABLE(lpmOctreeTest LpmOctreeTests.cpp)
TARGET_LINK_LIBRARIES(lpmOctreeTest lpm ${Trilinos_LIBRARIES} ${CMAKE_DL_LIBS} ${VTK_LIBRARIES} ${Trilinos_TPL_LIBRARIES})
#ADD_TEST(lpmOctreeTest lpmOctreeTest)

ADD_EXECUTABLE(lpmSphereVoronoiPrimitives LpmSphereVoronoiPrimitivesUnitTests.cpp)
TARGET_LINK_LIBRARIES(lpmSphereVoronoiPrimitives lpm ${Trilinos_LIBRARIES} ${CMAKE_DL_LIBS} ${VTK_LIBRARIES} ${Trilinos_TPL_LIBRARIES})
ADD_TEST(lpmSphereVoronoiPrimitives lpmSphereVoronoiPrimitives)

ADD_EXECUTABLE(lpmKokkosKernelsTest LpmKokkosKernelsTests.cpp)
TARGET_LINK_LIBRARIES(lpmKokkosKernelsTest lpm ${Trilinos_LIBRARIES} ${CMAKE_DL_LIBS} ${Trilinos_TPL_LIBRARIES})
ADD_TEST(lpmKokkosKernelsTest lpmKokkosKernelsTest)

#ADD_EXECUTABLE(lpmSphereVoronoiMesh LpmSphereVoronoiMeshUnitTests.cpp)
#TARGET_LINK_LIBRARIES(lpmSphereVoronoiMesh lpm ${Trilinos_LIBRARIES} ${CMAKE_DL_LIBS} ${VTK_LIBRARIES} ${NETCDF_C} ${NETCDF_CXX})
#ADD_TEST(lpmSphereVoronoiMesh lpmSphereVoronoiMesh)

ADD_EXECUTABLE(lpmBVETest LpmBVETest.cpp)
TARGET_LINK_LIBRARIES(lpmBVETest lpm ${Trilinos_LIBRARIES} ${CMAKE_DL_LIBS} ${VTK_LIBRARIES} ${Trilinos_TPL_LIBRARIES})
ADD_TEST(lpmBVETest lpmBVETest)

