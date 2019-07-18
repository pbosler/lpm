add_library(lpm LpmUtilities.cpp LpmCoords.cpp LpmMeshSeed.cpp LpmEdges.cpp LpmFaces.cpp LpmVtkIO.cpp 
    LpmPolyMesh2d.cpp LpmBVESphere.cpp)
ADD_DEPENDENCIES(lpm kokkos)
TARGET_LINK_LIBRARIES(lpm kokkos ${VTK_LIBRARIES})

install(FILES LpmConfig.h LpmDefs.hpp LpmUtilities.hpp LpmCoords.hpp LpmEdges.hpp LpmFaces.hpp LpmMeshSeed.hpp LpmVtkIO.hpp 
    DESTINATION include)
install(TARGETS lpm DESTINATION lib)
