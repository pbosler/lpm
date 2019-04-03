include (CTest)
INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/src" "${PROJECT_BINARY_DIR}" "${VTK_INCLUDE_DIRS}" "${Kokkos_ROOT}/include")

ADD_EXECUTABLE(lpmRealVecTest LpmRealVecUnitTest.cpp)
SET_TARGET_PROPERTIES(lpmRealVecTest PROPERTIES COMPILE_FLAGS "${LPM_CXXFLAGS}" LINK_FLAGS "${LPM_LDFLAGS}")
TARGET_LINK_LIBRARIES(lpmRealVecTest lpmkokkos)
add_test(lpmRealVectors lpmRealVecTest)

