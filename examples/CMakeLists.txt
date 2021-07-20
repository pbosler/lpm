add_executable(sphere_mesh_init sphere_mesh_init.cpp)
target_include_directories(sphere_mesh_init PUBLIC
        ${LPM_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${lpmtest_INCLUDE_DIRS}
        )
target_link_libraries(sphere_mesh_init lpm ${LPM_LIBRARIES} ${CMAKE_DL_LIBS})

#ADD_EXECUTABLE(spherePoisson SphereVorticity.cpp)
#TARGET_LINK_LIBRARIES(spherePoisson lpm ${Trilinos_LIBRARIES} ${VTK_LIBRARIES} ${Trilinos_TPL_LIBRARIES})
#
