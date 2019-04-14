INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/src" "${PROJECT_BINARY_DIR}" "${VTK_INCLUDE_DIRS}" "${Kokkos_ROOT}/include")

add_library(lpmkokkos LpmUtilities.cpp LpmCoords.cpp LpmMeshSeed.cpp LpmEdges.cpp LpmFaces.cpp LpmVtkIO.cpp)
SET_TARGET_PROPERTIES(lpmkokkos PROPERTIES COMPILE_FLAGS "${LPM_CXXFLAGS}" LINK_FLAGS "${LPM_LDFLAGS}")
TARGET_LINK_LIBRARIES(lpmkokkos ${Kokkos_LIBRARIES} ${VTK_LIBRARIES})

install(FILES LpmConfig.h LpmDefs.hpp LpmUtilities.hpp LpmCoords.hpp LpmEdges.hpp LpmFaces.hpp LpmMeshSeed.hpp LpmVtkIO.hpp
    DESTINATION include)
install(TARGETS lpmkokkos DESTINATION lib)
