INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/src" "${PROJECT_BINARY_DIR}" "${VTK_INCLUDE_DIRS}" "${Kokkos_ROOT}/include")

add_library(lpmkokkos LpmUtilities.cpp LpmRealVector.cpp)
SET_TARGET_PROPERTIES(lpmkokkos PROPERTIES COMPILE_FLAGS "${LPM_CXXFLAGS}" LINK_FLAGS "${LPM_LDFLAGS}")

install (FILES LpmConfig.h LpmTypeDefs.hpp LpmUtilities.hpp LpmRealVector.hpp
    DESTINATION include)
