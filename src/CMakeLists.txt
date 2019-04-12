INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/src" "${PROJECT_BINARY_DIR}" "${VTK_INCLUDE_DIRS}" "${Kokkos_ROOT}/include")

add_library(lpmkokkos LpmUtilities.cpp LpmCoords.cpp LpmEdges.cpp LpmFaces.cpp LpmSeedReader.cpp)
SET_TARGET_PROPERTIES(lpmkokkos PROPERTIES COMPILE_FLAGS "${LPM_CXXFLAGS}" LINK_FLAGS "${LPM_LDFLAGS}")
TARGET_LINK_LIBRARIES(lpmkokkos ${Kokkos_LIBRARIES})

install (FILES LpmConfig.h LpmDefs.hpp LpmUtilities.hpp LpmCoords.hpp LpmEdges.hpp LpmFaces.hpp LpmSeedReader.hpp
    DESTINATION include)
