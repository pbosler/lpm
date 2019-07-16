add_library(lpmkokkos LpmUtilities.cpp LpmCoords.cpp LpmMeshSeed.cpp LpmEdges.cpp LpmFaces.cpp LpmVtkIO.cpp 
    LpmPolyMesh2d.cpp LpmBVESphere.cpp)
TARGET_LINK_LIBRARIES(lpmkokkos ${KOKKOS_LIBRARIES} ${VTK_LIBRARIES})

install(FILES LpmConfig.h LpmDefs.hpp LpmUtilities.hpp LpmCoords.hpp LpmEdges.hpp LpmFaces.hpp LpmMeshSeed.hpp LpmVtkIO.hpp 
    DESTINATION include)
install(TARGETS lpmkokkos DESTINATION lib)
