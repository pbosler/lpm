enable_testing()
if (VTK_FOUND)
add_executable(sphere_mesh_init sphere_mesh_init.cpp)
target_include_directories(sphere_mesh_init PUBLIC
        ${LPM_INCLUDE_DIRS}
        ${Trilinos_INCLUDE_DIRS}
        ${Trilinos_TPL_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${VTK_INCLUDE_DIRS}
        ${lpmtest_INCLUDE_DIRS}
        )
target_link_libraries(sphere_mesh_init PUBLIC lpm ${LPM_LIBRARIES} ${LPM_EXT_LIBRARIES} ${VTK_LIBRARIES} ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} ${CMAKE_DL_LIBS})

add_executable(bve_rotation bve_rotation.cpp)
target_include_directories(bve_rotation PUBLIC
  ${LPM_INCLUDE_DIRS}
  ${Trilinos_INCLUDE_DIRS}
  ${Trilinos_TPL_INCLUDE_DIRS}
  ${VTK_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${lpmtest_INCLUDE_DIRS}
  )
target_link_libraries(bve_rotation PUBLIC lpm ${LPM_LIBRARIES} ${LPM_EXT_LIBRARIES} ${VTK_LIBRARIES} ${CMAKE_DL_LIBS})
else()
  message("VTK not found. Skipping examples:
      sphere_mesh_init
      bve_rotation")
endif()

add_test(NAME sphere_mesh_init
  COMMAND sh -c ${CMAKE_CURRENT_BINARY_DIR}/sphere_mesh_init -o sph_mesh_example.vtk -n 2)

add_test(NAME bve_solid_body_rotation
  COMMAND sh -c ${CMAKE_CURRENT_BINARY_DIR}/bve_rotation -o bve_test -d 3 -dt 0.01 -tf 0.03 -f 1)
