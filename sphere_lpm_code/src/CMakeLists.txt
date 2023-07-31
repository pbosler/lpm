


set(DFS_SOURCES
    dfs_doubling.cpp dfs_laplacian_new.cpp dfs_rhs_new.cpp dfs_solve_new.cpp
    dfs_velocity.cpp dfs_test_support.cpp dfs_interpolation.cpp 
    dfs_vort2velocity.cpp)

add_library(dfs ${DFS_SOURCES} )
target_include_directories(dfs PUBLIC ${PROJECT_SOURCE_DIR}/src ${FFTW3_INCLUDE_DIR} ${FFTW3THREADS_INCLUDE_DIR}
)
target_link_libraries(dfs PUBLIC Kokkos::kokkos  fftw3 fftw3_threads finufft)
