INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/src" "${PROJECT_BINARY_DIR}" "${VTK_INCLUDE_DIRS}" "${Kokkos_ROOT}/include")

#get_property(cxx_features GLOBAL PROPERTY CMAKE_CXX_KNOWN_FEATURES)
#set(CMAKE_CXX_COMPILE_FEATURES ${cxx_features})
#set(GLOBAL PROPERTY CMAKE_C_COMPILE_FEATURES ${cxx_features})

add_library(lpm LpmUtilities.cpp LpmCoords.cpp LpmMeshSeed.cpp LpmEdges.cpp LpmFaces.cpp LpmVtkIO.cpp 
    LpmPolyMesh2d.cpp LpmBVESphere.cpp)
#SET_TARGET_PROPERTIES(lpm PROPERTIES COMPILE_FLAGS "${LPM_CXXFLAGS}" LINK_FLAGS "${LPM_LDFLAGS}")
TARGET_LINK_LIBRARIES(lpm kokkos ${VTK_LIBRARIES})

install(FILES LpmConfig.h LpmDefs.hpp LpmUtilities.hpp LpmCoords.hpp LpmEdges.hpp LpmFaces.hpp LpmMeshSeed.hpp LpmVtkIO.hpp 
    DESTINATION include)
install(TARGETS lpm DESTINATION lib)
