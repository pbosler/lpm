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

add_executable(sphere_transport sphere_transport.cpp)
target_link_libraries(sphere_transport PUBLIC lpm ${LPM_LIBRARIES} ${LPM_EXT_LIBRARIES} ${VTK_LIBRARIES} ${CMAKE_DL_LIBS})
target_include_directories(sphere_transport PUBLIC
 ${LPM_INCLUDE_DIRS}
 ${Trilinos_INCLUDE_DIRS}
 ${Trilinos_TPL_INCLUDE_DIRS}
 ${VTK_INCLUDE_DIRS}
 ${CMAKE_CURRENT_SOURCE_DIR}
 ${lpmtest_INCLUDE_DIRS}
)

add_executable(plane_gravity_wave plane_gravity_wave.cpp)
target_link_libraries(plane_gravity_wave PUBLIC lpm ${LPM_LIBRARIES} ${LPM_EXT_LIBRARIES} ${VTK_LIBRARIES} ${CMAKE_DL_LIBS})
target_include_directories(plane_gravity_wave PUBLIC
 ${LPM_INCLUDE_DIRS}
 ${Trilinos_INCLUDE_DIRS}
 ${Trilinos_TPL_INCLUDE_DIRS}
 ${VTK_INCLUDE_DIRS}
 ${CMAKE_CURRENT_SOURCE_DIR}
 ${lpmtest_INCLUDE_DIRS}
)

add_executable(colliding_dipoles plane_colliding_dipoles.cpp)
target_link_libraries(colliding_dipoles PUBLIC lpm ${LPM_LIBRARIES} ${LPM_EXT_LIBRARIES} ${VTK_LIBRARIES} ${CMAKE_DL_LIBS})
target_include_directories(colliding_dipoles PUBLIC
 ${LPM_INCLUDE_DIRS}
 ${Trilinos_INCLUDE_DIRS}
 ${Trilinos_TPL_INCLUDE_DIRS}
 ${VTK_INCLUDE_DIRS}
 ${CMAKE_CURRENT_SOURCE_DIR}
 ${lpmtest_INCLUDE_DIRS}
)

add_executable(sphere_gaussian_vortex sphere_gaussian_vortex.cpp)
target_link_libraries(sphere_gaussian_vortex PUBLIC lpm ${LPM_LIBRARIES} ${LPM_EXT_LIBRARIES} ${VTK_LIBRARIES} ${CMAKE_DL_LIBS})
target_include_directories(sphere_gaussian_vortex PUBLIC
 ${LPM_INCLUDE_DIRS}
 ${Trilinos_INCLUDE_DIRS}
 ${Trilinos_TPL_INCLUDE_DIRS}
 ${VTK_INCLUDE_DIRS}
 ${CMAKE_CURRENT_SOURCE_DIR}
 ${lpmtest_INCLUDE_DIRS}
)

add_executable(sphere_rh54 sphere_rh54.cpp)
target_link_libraries(sphere_rh54 PUBLIC lpm ${LPM_LIBRARIES} ${LPM_EXT_LIBRARIES} ${VTK_LIBRARIES} ${CMAKE_DL_LIBS})
target_include_directories(sphere_rh54 PUBLIC
 ${LPM_INCLUDE_DIRS}
 ${Trilinos_INCLUDE_DIRS}
 ${Trilinos_TPL_INCLUDE_DIRS}
 ${VTK_INCLUDE_DIRS}
 ${CMAKE_CURRENT_SOURCE_DIR}
 ${lpmtest_INCLUDE_DIRS}
)


add_executable(sphere_swe_test_case2 sphere_swe_tc2.cpp)
target_link_libraries(sphere_swe_test_case2 PUBLIC lpm ${LPM_LIBRARIES} ${LPM_EXT_LIBRARIES} ${VTK_LIBRARIES} ${CMAKE_DL_LIBS})
target_include_directories(sphere_swe_test_case2 PUBLIC
 ${LPM_INCLUDE_DIRS}
 ${Trilinos_INCLUDE_DIRS}
 ${Trilinos_TPL_INCLUDE_DIRS}
 ${VTK_INCLUDE_DIRS}
 ${CMAKE_CURRENT_SOURCE_DIR}
 ${lpmtest_INCLUDE_DIRS}
)

#add_executable(sphere_staggered_swe_tc2 sphere_staggered_swe_tc2.cpp)
#target_link_libraries(sphere_staggered_swe_tc2 PUBLIC lpm ${LPM_LIBRARIES} ${LPM_EXT_LIBRARIES} ${VTK_LIBRARIES} ${CMAKE_DL_LIBS})
#target_include_directories(sphere_staggered_swe_tc2 PUBLIC
# ${LPM_INCLUDE_DIRS}
# ${Trilinos_INCLUDE_DIRS}
# ${Trilinos_TPL_INCLUDE_DIRS}
# ${VTK_INCLUDE_DIRS}
# ${CMAKE_CURRENT_SOURCE_DIR}
# ${lpmtest_INCLUDE_DIRS}
#)

add_executable(plane_swe_gaussians plane_swe_gaussian_tests.cpp)
target_link_libraries(plane_swe_gaussians PUBLIC lpm ${LPM_LIBRARIES} ${LPM_EXT_LIBRARIES} ${VTK_LIBRARIES} ${CMAKE_DL_LIBS})
target_include_directories(plane_swe_gaussians PUBLIC
 ${LPM_INCLUDE_DIRS}
 ${Trilinos_INCLUDE_DIRS}
 ${Trilinos_TPL_INCLUDE_DIRS}
 ${VTK_INCLUDE_DIRS}
 ${CMAKE_CURRENT_SOURCE_DIR}
 ${lpmtest_INCLUDE_DIRS}
)

add_executable(plane_swe_high_order plane_swe_high_order_tests.cpp)
target_link_libraries(plane_swe_high_order PUBLIC lpm ${LPM_LIBRARIES} ${LPM_EXT_LIBRARIES} ${VTK_LIBRARIES} ${CMAKE_DL_LIBS})
target_include_directories(plane_swe_high_order PUBLIC
 ${LPM_INCLUDE_DIRS}
 ${Trilinos_INCLUDE_DIRS}
 ${Trilinos_TPL_INCLUDE_DIRS}
 ${VTK_INCLUDE_DIRS}
 ${CMAKE_CURRENT_SOURCE_DIR}
 ${lpmtest_INCLUDE_DIRS}
)

if (LPM_ENABLE_NETCDF)
add_executable(sphere_from_data sphere_from_data.cpp)
target_link_libraries(sphere_from_data PUBLIC lpm ${LPM_LIBRARIES} ${LPM_EXT_LIBRARIES} ${VTK_LIBRARIES} ${CMAKE_DL_LIBS})
target_include_directories(sphere_from_data PUBLIC
 ${LPM_INCLUDE_DIRS}
 ${Trilinos_INCLUDE_DIRS}
 ${Trilinos_TPL_INCLUDE_DIRS}
 ${VTK_INCLUDE_DIRS}
 ${CMAKE_CURRENT_SOURCE_DIR}
 ${lpmtest_INCLUDE_DIRS}
)

add_test(NAME sphere_from_data
  COMMAND sphere_from_data ${PROJECT_SOURCE_DIR}/tests/data/so2_plume.nc -f so2)
endif()

if (LPM_ENABLE_DFS)
  add_executable(dfs_bve_rh54 dfs_bve_rh54.cpp)
  target_link_libraries(dfs_bve_rh54 PUBLIC lpm ${LPM_LIBRARIES} ${LPM_EXT_LIBRARIES} ${VTK_LIBRARIES} ${CMAKE_DL_LIBS})
  target_include_directories(dfs_bve_rh54 PUBLIC ${LPM_INCLUDE_DIRS} ${VTK_INCLUDE_DIRS} ${lpmtest_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(dfs_bve_gauss_vort dfs_bve_gauss_vort.cpp)
  target_link_libraries(dfs_bve_gauss_vort PUBLIC lpm ${LPM_LIBRARIES} ${LPM_EXT_LIBRARIES} ${VTK_LIBRARIES} ${CMAKE_DL_LIBS})
  target_include_directories(dfs_bve_gauss_vort PUBLIC ${LPM_INCLUDE_DIRS} ${VTK_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

add_test(NAME dfs_bve_rh54 COMMAND dfs_bve_rh54 -d 2 -nl 10 -dt 0.05 -tf 0.05)

endif(LPM_ENABLE_DFS)

add_test(NAME sphere_mesh_init
  COMMAND sphere_mesh_init -o sph_mesh_example.vtk -n 2)

add_test(NAME bve_solid_body_rotation
 COMMAND bve_rotation -o bve_test -d 3 -dt 0.01 -tf 0.03 -f 1)

add_test(NAME sphere_transport
 COMMAND sphere_transport -d 3 -dt 0.01 -tf 0.03 -of -1 -rf -1 -amr 1 -amr_max 1)

add_test(NAME colliding_dipoles_direct_unif
  COMMAND colliding_dipoles -d 3 -n 2 -tf 0.2 -rm 1 -of 20)

add_test(NAME colliding_dipoles_indirect_unif
  COMMAND colliding_dipoles -d 3 -n 2 -tf 0.2 -rm 1 -of 20 -rs "indirect")

add_test(NAME colliding_dipoles_indirect_amr
  COMMAND colliding_dipoles -d 3 -n 2 -tf 0.2 -amr 2 -c 0.1 -rm 1 -of 20 -rs "indirect")

add_test(NAME colliding_dipoles_direct_amr
  COMMAND colliding_dipoles -d 3 -n 2 -tf 0.2 -amr 2 -c 0.1 -rm 1 -of 20)

else()
  message("VTK not found. Skipping examples:
      sphere_mesh_init
      bve_rotation
      sphere_transport_amr")
endif()




