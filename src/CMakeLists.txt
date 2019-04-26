INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/src" "${PROJECT_BINARY_DIR}" "${VTK_INCLUDE_DIRS}" "${Kokkos_ROOT}/include")

get_property(cxx_features GLOBAL PROPERTY CMAKE_CXX_KNOWN_FEATURES)
set(CMAKE_CXX_COMPILE_FEATURES ${cxx_features})
set(GLOBAL PROPERTY CMAKE_C_COMPILE_FEATURES ${cxx_features})

add_library(lpmkokkos LpmUtilities.cpp LpmCoords.cpp LpmMeshSeed.cpp LpmEdges.cpp)# LpmFaces.cpp LpmVtkIO.cpp LpmPolyMesh2d.cpp)
SET_TARGET_PROPERTIES(lpmkokkos PROPERTIES COMPILE_FLAGS "${LPM_CXXFLAGS}" LINK_FLAGS "${LPM_LDFLAGS}")
TARGET_LINK_LIBRARIES(lpmkokkos ${Kokkos_LIBRARIES} ${VTK_LIBRARIES})

install(FILES LpmConfig.h LpmDefs.hpp LpmUtilities.hpp LpmCoords.hpp LpmEdges.hpp LpmFaces.hpp LpmMeshSeed.hpp LpmVtkIO.hpp 
    DESTINATION include)
install(TARGETS lpmkokkos DESTINATION lib)
