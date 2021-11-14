if (VTK_FOUND)
add_executable(sphere_mesh_init sphere_mesh_init.cpp)
target_include_directories(sphere_mesh_init PUBLIC
        ${LPM_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${VTK_INCLUDE_DIRS}
        ${lpmtest_INCLUDE_DIRS}
        )
target_link_libraries(sphere_mesh_init PUBLIC lpm ${LPM_LIBRARIES} ${LPM_EXT_LIBRARIES} ${VTK_LIBRARIES})

add_executable(bve_rotation bve_rotation.cpp)
target_include_directories(bve_rotation PUBLIC
  ${LPM_INCLUDE_DIRS}
  ${VTK_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${lpmtest_INCLUDE_DIRS}
  )
target_link_libraries(bve_rotation PUBLIC lpm ${LPM_LIBRARIES} ${LPM_EXT_LIBRARIES} ${VTK_LIBRARIES})
else()
  message("VTK not found. Skipping examples:
      sphere_mesh_init
      bve_rotation")
endif()
