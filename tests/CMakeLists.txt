include (CTest)

#printvar(kokkos)

get_property(cxx_features GLOBAL PROPERTY CMAKE_CXX_KNOWN_FEATURES)
set(CMAKE_CXX_COMPILE_FEATURES ${cxx_features})
set(GLOBAL PROPERTY CMAKE_C_COMPILE_FEATURES ${cxx_features})

ADD_EXECUTABLE(lpmGeomTest LpmGeometryUnitTests.cpp)
TARGET_LINK_LIBRARIES(lpmGeomTest lpm ${KOKKOS_LIBRARIES} ${VTK_LIBRARIES})
ADD_TEST(lpmGeom lpmGeomTest)

ADD_EXECUTABLE(lpmCoordsTest LpmCoordsUnitTest.cpp)
TARGET_LINK_LIBRARIES(lpmCoordsTest lpm ${KOKKOS_LIBRARIES} ${VTK_LIBRARIES})
ADD_TEST(lpmCoords lpmCoordsTest)

ADD_EXECUTABLE(lpmEdgesTest LpmEdgesUnitTest.cpp)
TARGET_LINK_LIBRARIES(lpmEdgesTest lpm ${KOKKOS_LIBRARIES} ${VTK_LIBRARIES})
ADD_TEST(lpmEdges lpmEdgesTest)

ADD_EXECUTABLE(lpmFacesTest LpmFacesUnitTests.cpp)
TARGET_LINK_LIBRARIES(lpmFacesTest lpm ${KOKKOS_LIBRARIES} ${VTK_LIBRARIES})
ADD_TEST(lpmFaces lpmFacesTest)

ADD_EXECUTABLE(lpmMeshSeedTest LpmMeshSeedUnitTest.cpp)
TARGET_LINK_LIBRARIES(lpmMeshSeedTest lpm ${KOKKOS_LIBRARIES} ${VTK_LIBRARIES})
ADD_TEST(lpmMeshSeed lpmMeshSeedTest)

ADD_EXECUTABLE(lpmPolyMeshTest LpmPolyMeshTest.cpp)
TARGET_LINK_LIBRARIES(lpmPolyMeshTest lpm ${KOKKOS_LIBRARIES} ${VTK_LIBRARIES})
ADD_TEST(lpmPolyMeshTest lpmPolyMeshTest)

ADD_EXECUTABLE(lpmKernelTest LpmKernelTest.cpp)
TARGET_LINK_LIBRARIES(lpmKernelTest lpm ${KOKKOS_LIBRARIES} ${VTK_LIBRARIES})
ADD_TEST(lpmKernelTest lpmKernelTest)

ADD_EXECUTABLE(lpmSpherePoissonTest LpmSpherePoisson.cpp)
TARGET_LINK_LIBRARIES(lpmSpherePoissonTest lpm ${KOKKOS_LIBRARIES} ${VTK_LIBRARIES})
ADD_TEST(lpmSpherePoissonTest lpmSpherePoissonTest)
#
#ADD_EXECUTABLE(lpmBVETest LpmBVETest.cpp)
#TARGET_LINK_LIBRARIES(lpmBVETest lpm ${KOKKOS_LIBRARIES} ${VTK_LIBRARIES})
#ADD_TEST(lpmBVETest lpmBVETest)

