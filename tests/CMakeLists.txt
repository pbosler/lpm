include (CTest)
INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/src"
                    "${PROJECT_BINARY_DIR}" 
                    "${VTK_INCLUDE_DIRS}" 
                    "${Kokkos_ROOT}/include" 
                    "${Compadre_ROOT}/include"
)

#get_property(cxx_features GLOBAL PROPERTY CMAKE_CXX_KNOWN_FEATURES)
#set(CMAKE_CXX_COMPILE_FEATURES ${cxx_features})
#set(GLOBAL PROPERTY CMAKE_C_COMPILE_FEATURES ${cxx_features})

ADD_EXECUTABLE(lpmGeomTest LpmGeometryUnitTests.cpp)
#SET_TARGET_PROPERTIES(lpmGeomTest PROPERTIES COMPILE_FLAGS "${LPM_CXXFLAGS}" LINK_FLAGS "${LPM_LDFLAGS}")
TARGET_LINK_LIBRARIES(lpmGeomTest lpm kokkos ${VTK_LIBRARIES})
ADD_TEST(lpmGeom lpmGeomTest)

ADD_EXECUTABLE(lpmCoordsTest LpmCoordsUnitTest.cpp)
#SET_TARGET_PROPERTIES(lpmCoordsTest PROPERTIES COMPILE_FLAGS "${LPM_CXXFLAGS}" LINK_FLAGS "${LPM_LDFLAGS}")
TARGET_LINK_LIBRARIES(lpmCoordsTest lpm kokkos ${VTK_LIBRARIES})
ADD_TEST(lpmCoords lpmCoordsTest)

ADD_EXECUTABLE(lpmEdgesTest LpmEdgesUnitTest.cpp)
#SET_TARGET_PROPERTIES(lpmEdgesTest PROPERTIES COMPILE_FLAGS "${LPM_CXXFLAGS}" LINK_FLAGS "${LPM_LDFLAGS}")
TARGET_LINK_LIBRARIES(lpmEdgesTest lpm kokkos ${VTK_LIBRARIES})
ADD_TEST(lpmEdges lpmEdgesTest)

ADD_EXECUTABLE(lpmFacesTest LpmFacesUnitTests.cpp)
#SET_TARGET_PROPERTIES(lpmFacesTest PROPERTIES COMPILE_FLAGS "${LPM_CXXFLAGS}" LINK_FLAGS "${LPM_LDFLAGS}")
TARGET_LINK_LIBRARIES(lpmFacesTest lpm kokkos ${VTK_LIBRARIES})
ADD_TEST(lpmFaces lpmFacesTest)

ADD_EXECUTABLE(lpmMeshSeedTest LpmMeshSeedUnitTest.cpp)
#SET_TARGET_PROPERTIES(lpmMeshSeedTest PROPERTIES COMPILE_FLAGS "${LPM_CXXFLAGS}" LINK_FLAGS "${LPM_LDFLAGS}")
TARGET_LINK_LIBRARIES(lpmMeshSeedTest lpm kokkos ${VTK_LIBRARIES})
ADD_TEST(lpmMeshSeed lpmMeshSeedTest)

ADD_EXECUTABLE(lpmPolyMeshTest LpmPolyMeshTest.cpp)
#SET_TARGET_PROPERTIES(lpmPolyMeshTest PROPERTIES COMPILE_FLAGS "${LPM_CXXFLAGS}" LINK_FLAGS "${LPM_LDFLAGS}")
TARGET_LINK_LIBRARIES(lpmPolyMeshTest lpm kokkos ${VTK_LIBRARIES})
ADD_TEST(lpmPolyMeshTest lpmPolyMeshTest)

ADD_EXECUTABLE(lpmKernelTest LpmKernelTest.cpp)
SET_TARGET_PROPERTIES(lpmKernelTest PROPERTIES COMPILE_FLAGS "${LPM_CXXFLAGS}" LINK_FLAGS "${LPM_LDFLAGS}")
TARGET_LINK_LIBRARIES(lpmKernelTest lpm kokkos ${VTK_LIBRARIES})
ADD_TEST(lpmKernelTest lpmKernelTest)

ADD_EXECUTABLE(lpmSpherePoissonTest LpmSpherePoisson.cpp)
#SET_TARGET_PROPERTIES(lpmSpherePoissonTest PROPERTIES COMPILE_FLAGS "${LPM_CXXFLAGS}" LINK_FLAGS "${LPM_LDFLAGS}")
TARGET_LINK_LIBRARIES(lpmSpherePoissonTest lpm kokkos ${VTK_LIBRARIES})
ADD_TEST(lpmSpherePoissonTest lpmSpherePoissonTest)

#ADD_EXECUTABLE(lpmGMLS LpmGMLSTest.cpp)
##SET_TARGET_PROPERTIES(lpmGMLS PROPERTIES COMPILE_FLAGS "${LPM_CXXFLAGS}" LINK_FLAGS "${LPM_LDFLAGS}")
#TARGET_LINK_LIBRARIES(lpmGMLS lpm compadre kokkos ${VTK_LIBRARIES} ${CMAKE_DL_LIBS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} )
#ADD_TEST(lpmGMLS lpmGMLS)

ADD_EXECUTABLE(lpmLatLonMesh LpmLatLonMeshTests.cpp)
TARGET_LINK_LIBRARIES(lpmLatLonMesh lpm kokkos ${VTK_LIBRARIES} ${CMAKE_DL_LIBS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
ADD_TEST(lpmLatLonMeshTest lpmLatLonMesh)

ADD_EXECUTABLE(lpmCompadre LpmCompadreTests.cpp)
TARGET_LINK_LIBRARIES(lpmCompadre lpm compadre kokkos ${VTK_LIBRARIES} ${CMAKE_DL_LIBS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
if (USE_CUDA) 
   TARGET_LINK_LIBRARIES(lpmCompadre cublas cudart cusolver)
endif()
ADD_TEST(lpmCompadreTest lpmCompadre)

#ADD_EXECUTABLE(lpmGaussGrid LpmGaussGridTests.cpp)
#TARGET_LINK_LIBRARIES(lpmGaussGrid lpm compadre kokkos ${VTK_LIBRARIES} ${CMAKE_DL_LIBS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
#ADD_TEST(lpmGaussGridTest lpmGaussGrid)

#ADD_EXECUTABLE(lpmOctreeUtilsTest LpmOctreeUtilsUnitTests.cpp)
#TARGET_LINK_LIBRARIES(lpmOctreeUtilsTest lpm kokkos ${CMAKE_DL_LIBS})
#ADD_TEST(lpmOctreeUtilsTest lpmOctreeUtilsTest)
#
#ADD_EXECUTABLE(lpmBVETest LpmBVETest.cpp)
#SET_TARGET_PROPERTIES(lpmBVETest PROPERTIES COMPILE_FLAGS "${LPM_CXXFLAGS}" LINK_FLAGS "${LPM_LDFLAGS}")
#TARGET_LINK_LIBRARIES(lpmBVETest lpm kokkos ${VTK_LIBRARIES})
#ADD_TEST(lpmBVETest lpmBVETest)

