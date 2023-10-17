include(create_unit_test)
enable_testing()

#######################
###    THREADING    ###
#######################

if (LPM_USE_CUDA)
  set (DEFAULT_MAX_THREADS 1)
else ()
  set (DEFAULT_MAX_THREADS 16)
endif ()

set (LPM_TEST_MAX_RANKS 4 CACHE STRING "Max number of ranks for tests")
set (LPM_TEST_MAX_THREADS ${DEFAULT_MAX_THREADS} CACHE STRING "Max number of threads for tests")
set (LPM_TEST_THREAD_INC 1  CACHE STRING "Increment for threads for tests")

CreateUnitTest(geometry_test lpm_geometry_tests.cpp)

CreateUnitTest(mpi_comm_test lpm_comm_tests.cpp
  MPI_RANKS 1 ${LPM_TEST_MAX_RANKS}
  MPI_EXTRA_ARGS "--oversubscribe")

CreateUnitTest(logger_test lpm_logger_tests.cpp)

CreateUnitTest(mpi_logger mpi_log_rank0_only.cpp
  MPI_RANKS 1 ${LPM_TEST_MAX_RANKS}
  MPI_EXTRA_ARGS "--oversubscribe")

CreateUnitTest(multicomponent_logger multicomponent_log_test.cpp
  MPI_RANKS 1 ${LPM_TEST_MAX_RANKS}
  MPI_EXTRA_ARGS "--oversubscribe")

CreateUnitTest(mesh_seed lpm_mesh_seed_tests.cpp)

CreateUnitTest(coords_test lpm_coords_test.cpp)

CreateUnitTest(math_test lpm_math_tests.cpp)

CreateUnitTest(vertices_test lpm_vertices_tests.cpp)

CreateUnitTest(edges_test lpm_edges_tests.cpp)

CreateUnitTest(faces_test lpm_faces_tests.cpp)

CreateUnitTest(polymesh_test lpm_polymesh_tests.cpp)

CreateUnitTest(box3d_test lpm_box3d_tests.cpp)

CreateUnitTest(tracer_init_test lpm_tracer_init_test.cpp)

CreateUnitTest(error_unit_test lpm_error_unit_tests.cpp)

CreateUnitTest(transport_2d_unit_test lpm_2d_transport_tests.cpp)

CreateUnitTest(pse_unit_test lpm_pse_tests.cpp)

CreateUnitTest(polymesh_functions lpm_polymesh2d_function_tests.cpp)

CreateUnitTest(latlon_unit_tests lpm_lat_lon_unit_tests.cpp)

CreateUnitTest(compadre_unit_tests lpm_compadre_unit_tests.cpp)

CreateUnitTest(gather_scatter lpm_gather_scatter_mesh_tests.cpp)

CreateUnitTest(bivar lpm_bivar_tests.cpp)

CreateUnitTest(ssrfpack lpm_ssrfpack_tests.cpp)

CreateUnitTest(swe_kernels lpm_swe_kernels_tests.cpp)

CreateUnitTest(swe_init lpm_swe_init_test.cpp)

CreateUnitTest(test_stub lpm_test_stub.cpp)

CreateUnitTest(netcdf_test lpm_netcdf_tests.cpp)

#ADD_EXECUTABLE(lpmKernelTest LpmKernelTest.cpp)
#TARGET_LINK_LIBRARIES(lpmKernelTest lpm ${Trilinos_LIBRARIES} ${VTK_LIBRARIES} ${Trilinos_TPL_LIBRARIES})
#ADD_TEST(lpmKernelTest lpmKernelTest)
#
#ADD_EXECUTABLE(lpmSpherePoissonTest LpmSpherePoisson.cpp)
#TARGET_LINK_LIBRARIES(lpmSpherePoissonTest lpm ${Trilinos_LIBRARIES} ${VTK_LIBRARIES} ${Trilinos_TPL_LIBRARIES})
#ADD_TEST(lpmSpherePoissonTest lpmSpherePoissonTest)
#
##ADD_EXECUTABLE(lpmGMLS LpmGMLSTest.cpp)
###SET_TARGET_PROPERTIES(lpmGMLS PROPERTIES COMPILE_FLAGS "${LPM_CXXFLAGS}" LINK_FLAGS "${LPM_LDFLAGS}")
##TARGET_LINK_LIBRARIES(lpmGMLS lpm compadre ${Trilinos_LIBRARIES} ${VTK_LIBRARIES} ${CMAKE_DL_LIBS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} )
##ADD_TEST(lpmGMLS lpmGMLS)
#
#ADD_EXECUTABLE(lpmLatLonMesh LpmLatLonMeshTests.cpp)
#TARGET_LINK_LIBRARIES(lpmLatLonMesh lpm ${Trilinos_LIBRARIES} ${VTK_LIBRARIES} ${CMAKE_DL_LIBS}  ${Trilinos_TPL_LIBRARIES})
#ADD_TEST(lpmLatLonMeshTest lpmLatLonMesh)
#
#ADD_EXECUTABLE(lpmCompadre LpmCompadreTests.cpp)
#TARGET_LINK_LIBRARIES(lpmCompadre lpm ${Trilinos_LIBRARIES} ${VTK_LIBRARIES} ${CMAKE_DL_LIBS} ${Trilinos_TPL_LIBRARIES})
#ADD_TEST(lpmCompadreTest lpmCompadre)
#
##ADD_EXECUTABLE(lpmGaussGrid LpmGaussGridTests.cpp)
##TARGET_LINK_LIBRARIES(lpmGaussGrid lpm compadre ${Trilinos_LIBRARIES} ${VTK_LIBRARIES} ${CMAKE_DL_LIBS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
##ADD_TEST(lpmGaussGridTest lpmGaussGrid)
#
#ADD_EXECUTABLE(lpmOctreeKeyTests LpmOctreeKeyTests.cpp)
#TARGET_LINK_LIBRARIES(lpmOctreeKeyTests lpm ${Trilinos_LIBRARIES} ${CMAKE_DL_LIBS} ${Trilinos_TPL_LIBRARIES})
#ADD_TEST(lpmOctreeKeyTests lpmOctreeKeyTests)
#
#ADD_EXECUTABLE(lpmOctreeLUTUnitTest LpmOctreeLUTUnitTests.cpp)
#TARGET_LINK_LIBRARIES(lpmOctreeLUTUnitTest lpm ${Trilinos_LIBRARIES} ${CMAKE_DL_LIBS} ${Trilinos_TPL_LIBRARIES})
#ADD_TEST(lpmOctreeLUTUnitTest lpmOctreeLUTUnitTest)
#
#ADD_EXECUTABLE(lpmOctreeUtilsTest LpmOctreeUtilsUnitTests.cpp)
#TARGET_LINK_LIBRARIES(lpmOctreeUtilsTest lpm ${Trilinos_LIBRARIES} ${CMAKE_DL_LIBS} ${Trilinos_TPL_LIBRARIES})
#ADD_TEST(lpmOctreeUtilsTest lpmOctreeUtilsTest)
#
#ADD_EXECUTABLE(lpmOctreeKernelTests LpmOctreeKernelTests.cpp)
#TARGET_LINK_LIBRARIES(lpmOctreeKernelTests lpm ${Trilinos_LIBRARIES} ${CMAKE_DL_LIBS} ${Trilinos_TPL_LIBRARIES})
#ADD_TEST(lpmOctreeKernelTests lpmOctreeKernelTests)
#
#ADD_EXECUTABLE(lpmNodeArrayDTest LpmNodeArrayDUnitTests.cpp)
#TARGET_LINK_LIBRARIES(lpmNodeArrayDTest lpm ${Trilinos_LIBRARIES} ${CMAKE_DL_LIBS} ${Trilinos_TPL_LIBRARIES})
#ADD_TEST(lpmNodeArrayDTest lpmNodeArrayDTest)
#
#ADD_EXECUTABLE(lpmNodeArrayInternalTest LpmNodeArrayInternalTests.cpp)
#TARGET_LINK_LIBRARIES(lpmNodeArrayInternalTest lpm ${Trilinos_LIBRARIES} ${CMAKE_DL_LIBS} ${Trilinos_TPL_LIBRARIES})
#ADD_TEST(lpmNodeArrayInternalTest lpmNodeArrayInternalTest)
#
#ADD_EXECUTABLE(lpmOctreeTest LpmOctreeTests.cpp)
#TARGET_LINK_LIBRARIES(lpmOctreeTest lpm ${Trilinos_LIBRARIES} ${CMAKE_DL_LIBS} ${VTK_LIBRARIES} ${Trilinos_TPL_LIBRARIES})
##ADD_TEST(lpmOctreeTest lpmOctreeTest)
#
#ADD_EXECUTABLE(lpmSphereVoronoiPrimitives LpmSphereVoronoiPrimitivesUnitTests.cpp)
#TARGET_LINK_LIBRARIES(lpmSphereVoronoiPrimitives lpm ${Trilinos_LIBRARIES} ${CMAKE_DL_LIBS} ${VTK_LIBRARIES} ${Trilinos_TPL_LIBRARIES})
#ADD_TEST(lpmSphereVoronoiPrimitives lpmSphereVoronoiPrimitives)
#
#ADD_EXECUTABLE(lpmKokkosKernelsTest LpmKokkosKernelsTests.cpp)
#TARGET_LINK_LIBRARIES(lpmKokkosKernelsTest lpm ${Trilinos_LIBRARIES} ${CMAKE_DL_LIBS} ${Trilinos_TPL_LIBRARIES})
#ADD_TEST(lpmKokkosKernelsTest lpmKokkosKernelsTest)
#
##ADD_EXECUTABLE(lpmSphereVoronoiMesh LpmSphereVoronoiMeshUnitTests.cpp)
##TARGET_LINK_LIBRARIES(lpmSphereVoronoiMesh lpm ${Trilinos_LIBRARIES} ${CMAKE_DL_LIBS} ${VTK_LIBRARIES} ${NETCDF_C} ${NETCDF_CXX})
##ADD_TEST(lpmSphereVoronoiMesh lpmSphereVoronoiMesh)
#
#ADD_EXECUTABLE(lpmBVETest LpmBVETest.cpp)
#TARGET_LINK_LIBRARIES(lpmBVETest lpm ${Trilinos_LIBRARIES} ${CMAKE_DL_LIBS} ${VTK_LIBRARIES} ${Trilinos_TPL_LIBRARIES})
#ADD_TEST(lpmBVETest lpmBVETest)
#
##ADD_EXECUTABLE(lpmNetCDFTest LpmNetCDFTest.cpp)
##TARGET_LINK_LIBRARIES(lpmNetCDFTest lpm ${Trilinos_LIBRARIES} ${CMAKE_DL_LIBS}
##  ${VTK_LIBRARIES} ${Trilinos_TPL_LIBRARIES} ${NETCDF_C} ${NETCDF_CXX})
##ADD_TEST(lpmNetCDFTest lpmNetCDFTest)
#
#ADD_EXECUTABLE(lpmPSEPlaneTest LpmPlanePSETest.cpp)
#TARGET_LINK_LIBRARIES(lpmPSEPlaneTest lpm ${Trilinos_LIBRARIES} ${CMAKE_DL_LIBS} ${VTK_LIBRARIES} ${Trilinos_TPL_LIBRARIES})
#ADD_TEST(lpmPSEPlaneTest lpmPSEPlaneTest)
#
#ADD_EXECUTABLE(lpmSWEPlaneTest LpmPlaneSWETest.cpp)
#TARGET_LINK_LIBRARIES(lpmSWEPlaneTest lpm ${Trilinos_LIBRARIES} ${CMAKE_DL_LIBS} ${VTK_LIBRARIES} ${Trilinos_TPL_LIBRARIES})
#ADD_TEST(lpmSWEPlaneTest lpmSWEPlaneTest)
