#INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/src"
#                    "${PROJECT_BINARY_DIR}"
#                    "${VTK_INCLUDE_DIRS}"
##                   "${Trilinos_INCLUDE_DIRS}"
##                   "${Trilinos_TPL_INCLUDE_DIRS}"
#)

add_executable(sphere_mesh_init sphere_mesh_init.cpp)
target_include_directories(sphere_mesh_init PUBLIC
        ${LPM_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        )
target_link_libraries(sphere_mesh_init PUBLIC lpm ${LPM_LIBRARIES} ${LPM_EXT_LIBRARIES})

#ADD_EXECUTABLE(spherePoisson SphereVorticity.cpp)
#TARGET_LINK_LIBRARIES(spherePoisson lpm ${Trilinos_LIBRARIES} ${VTK_LIBRARIES} ${Trilinos_TPL_LIBRARIES})
#
