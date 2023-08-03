
add_executable(dfs_double_test dfs_doubling_test.cpp)
target_include_directories(dfs_double_test PUBLIC ${PROJECT_SOURCE_DIR}/src ${FFTW3_INCLUDE_DIR} ${FFTW3THREADS_INCLUDE_DIR}
)
target_link_libraries(dfs_double_test PUBLIC dfs fftw3 fftw3_threads "-Wl,-no_compact_unwind")

add_test(dfs_double_test dfs_double_test)

add_executable(tridiag_solve_test dfs_tridiag_solve_test.cpp)
target_include_directories(tridiag_solve_test PUBLIC ${PROJECT_SOURCE_DIR}/src ${FFTW3_INCLUDE_DIR} ${FFTW3THREADS_INCLUDE_DIR}
)
target_link_libraries(tridiag_solve_test PUBLIC dfs fftw3 fftw3_threads "-Wl,-no_compact_unwind")

add_test(tridiag_solver_test tridiag_solve_test)

# Test the routine for creating laplace operator
add_executable(dfs_laplacian_test dfs_laplacian_test.cpp)
target_include_directories(dfs_laplacian_test PUBLIC ${PROJECT_SOURCE_DIR}/src ${FFTW3_INCLUDE_DIR} ${FFTW3THREADS_INCLUDE_DIR}
)
target_link_libraries(dfs_laplacian_test PUBLIC dfs fftw3 fftw3_threads "-Wl,-no_compact_unwind")

add_test(dfs_laplacian_test dfs_laplacian_test)

# Test routine for creating helmhotz operator
add_executable(dfs_helmholtz_test dfs_helmholtz_test.cpp)
target_include_directories(dfs_helmholtz_test PUBLIC ${PROJECT_SOURCE_DIR}/src ${FFTW3_INCLUDE_DIR} ${FFTW3THREADS_INCLUDE_DIR}
)
target_link_libraries(dfs_helmholtz_test PUBLIC dfs fftw3 fftw3_threads "-Wl,-no_compact_unwind")

add_test(dfs_helmholtz_test dfs_helmholtz_test)


# Test routine for computing Fourier coefficients
add_executable(dfs_vals2coeffs_test dfs_vals2coeffs_test.cpp)
target_include_directories(dfs_vals2coeffs_test PUBLIC ${PROJECT_SOURCE_DIR}/src ${FFTW3_INCLUDE_DIR} ${FFTW3THREADS_INCLUDE_DIR}
)
target_link_libraries(dfs_vals2coeffs_test PUBLIC dfs fftw3 fftw3_threads "-Wl,-no_compact_unwind")

add_test(dfs_vals2coeffs_test dfs_vals2coeffs_test)


# Test routine for computing values for Fourier coefficients
add_executable(dfs_coeffs2vals_test dfs_coeffs2vals_test.cpp)
target_include_directories(dfs_coeffs2vals_test PUBLIC ${PROJECT_SOURCE_DIR}/src ${FFTW3_INCLUDE_DIR} ${FFTW3THREADS_INCLUDE_DIR}
)
target_link_libraries(dfs_coeffs2vals_test PUBLIC dfs fftw3 fftw3_threads "-Wl,-no_compact_unwind")

add_test(dfs_coeffs2vals_test dfs_coeffs2vals_test)

# Test routine for discretizing the right handside of the Poisson
# Equation
add_executable(dfs_rhs_test dfs_rhs_test.cpp)
target_include_directories(dfs_rhs_test PUBLIC ${PROJECT_SOURCE_DIR}/src ${FFTW3_INCLUDE_DIR} ${FFTW3THREADS_INCLUDE_DIR}
)
target_link_libraries(dfs_rhs_test PUBLIC dfs fftw3 fftw3_threads "-Wl,-no_compact_unwind")

add_test(dfs_rhs_test dfs_rhs_test)

# Test the integration of the functions to solve Poisson
# Equation
add_executable(dfs_poisson_test dfs_poisson_test.cpp)
target_include_directories(dfs_poisson_test PUBLIC ${PROJECT_SOURCE_DIR}/src ${FFTW3_INCLUDE_DIR} ${FFTW3THREADS_INCLUDE_DIR}
)
target_link_libraries(dfs_poisson_test PUBLIC dfs fftw3 fftw3_threads "-Wl,-no_compact_unwind")

add_test(dfs_poisson_test dfs_poisson_test)


# Test the integration of the routines to solve the Helmholtz
# Equation
add_executable(dfs_Hsolver_test dfs_Hsolver_test.cpp)
target_include_directories(dfs_Hsolver_test PUBLIC ${PROJECT_SOURCE_DIR}/src ${FFTW3_INCLUDE_DIR} ${FFTW3THREADS_INCLUDE_DIR}
)
target_link_libraries(dfs_Hsolver_test PUBLIC dfs fftw3 fftw3_threads "-Wl,-no_compact_unwind")

add_test(dfs_Hsolver_test dfs_Hsolver_test)

# Test differentiation in Fourier space

add_executable(dfs_derivative_test dfs_derivative_test.cpp)
target_include_directories(dfs_derivative_test PUBLIC ${PROJECT_SOURCE_DIR}/src ${FFTW3_INCLUDE_DIR} ${FFTW3THREADS_INCLUDE_DIR}
)
target_link_libraries(dfs_derivative_test PUBLIC dfs fftw3 fftw3_threads "-Wl,-no_compact_unwind")

add_test(dfs_derivative_test dfs_derivative_test)

# Test the multiplication  of function by cos2theta
# Fourier space
add_executable(dfs_multi_test dfs_multi_cos2theta.cpp)
target_include_directories(dfs_multi_test PUBLIC ${PROJECT_SOURCE_DIR}/src ${FFTW3_INCLUDE_DIR} ${FFTW3THREADS_INCLUDE_DIR}
)
target_link_libraries(dfs_multi_test PUBLIC dfs fftw3 fftw3_threads "-Wl,-no_compact_unwind")

add_test(dfs_multiplycos2theta_test dfs_multi_test)


# Test the multiplication  of function by cos(lambda)/sin(lambda)
# Fourier space
add_executable(dfs_mattrig_test dfs_mattrig_test.cpp)
target_include_directories(dfs_mattrig_test PUBLIC ${PROJECT_SOURCE_DIR}/src ${FFTW3_INCLUDE_DIR} ${FFTW3THREADS_INCLUDE_DIR}
)
target_link_libraries(dfs_mattrig_test PUBLIC dfs fftw3 fftw3_threads "-Wl,-no_compact_unwind")

add_test(dfs_mattrig_test dfs_mattrig_test)

# Multiply a function by cos(theta)
add_executable(dfs_multiplybycos_test dfs_multicos_test.cpp)
target_include_directories(dfs_multiplybycos_test PUBLIC ${PROJECT_SOURCE_DIR}/src ${FFTW3_INCLUDE_DIR} ${FFTW3THREADS_INCLUDE_DIR}
)
target_link_libraries(dfs_multiplybycos_test PUBLIC dfs fftw3 fftw3_threads "-Wl,-no_compact_unwind")

add_test(dfs_multiplybycos_test dfs_multiplybycos_test)

# Divide a function by sin(theta)
add_executable(dfs_divsin_test dfs_divsin_test.cpp)
target_include_directories(dfs_divsin_test PUBLIC ${PROJECT_SOURCE_DIR}/src ${FFTW3_INCLUDE_DIR} ${FFTW3THREADS_INCLUDE_DIR}
)
target_link_libraries(dfs_divsin_test PUBLIC  dfs fftw3 fftw3_threads "-Wl,-no_compact_unwind")

add_test(dfs_divsin_test dfs_divsin_test)

# Test that velocity computed correctly on the grid
# from the stream function 
add_executable(dfs_gridv_test dfs_gridv_test.cpp)
target_include_directories(dfs_gridv_test PUBLIC ${PROJECT_SOURCE_DIR}/src ${FFTW3_INCLUDE_DIR} ${FFTW3THREADS_INCLUDE_DIR}
)
target_link_libraries(dfs_gridv_test PUBLIC  dfs fftw3 fftw3_threads "-Wl,-no_compact_unwind")

add_test(dfs_velocity_on_grid_test dfs_gridv_test)

# Test the use of finufft
# to interpolate
add_executable(dfs_finufft_test dfs_finufft_test.cpp)
target_include_directories(dfs_finufft_test PUBLIC ${PROJECT_SOURCE_DIR}/src ${FFTW3_INCLUDE_DIR} ${FFTW3THREADS_INCLUDE_DIR}
)
target_link_libraries(dfs_finufft_test PUBLIC finufft  dfs fftw3 fftw3_threads "-Wl,-no_compact_unwind")

add_test(dfs_finufft_test dfs_finufft_test)

# Test computing the velocity at particles given
# vorticity at colatatitude, longitude grid on the sphere
add_executable(dfs_vort2U_test dfs_vort2velocity_test.cpp)
target_include_directories(dfs_vort2U_test PUBLIC ${PROJECT_SOURCE_DIR}/src ${FFTW3_INCLUDE_DIR} ${FFTW3THREADS_INCLUDE_DIR}
)
target_link_libraries(dfs_vort2U_test PUBLIC finufft  dfs fftw3 fftw3_threads "-Wl,-no_compact_unwind")

add_test(dfs_vorticity_2_velocity_test dfs_vort2U_test)

# Test division by sine
# Test computing the velocity at particles given
# vorticity at colatatitude, longitude grid on the sphere
add_executable(dfs_divcos_test dfs_divcos_test.cpp)
target_include_directories(dfs_divcos_test PUBLIC ${PROJECT_SOURCE_DIR}/src ${FFTW3_INCLUDE_DIR} ${FFTW3THREADS_INCLUDE_DIR}
)
target_link_libraries(dfs_divcos_test PUBLIC finufft  dfs fftw3 fftw3_threads "-Wl,-no_compact_unwind")

add_test(dfs_divcos_test dfs_divcos_test)

# Test for  function computing 
# the surface gradient on the sphere
add_executable(dfs_surfgrad_test dfs_surfgrad_test.cpp)
target_include_directories(dfs_surfgrad_test PUBLIC ${PROJECT_SOURCE_DIR}/src ${FFTW3_INCLUDE_DIR} ${FFTW3THREADS_INCLUDE_DIR}
)
target_link_libraries(dfs_surfgrad_test PUBLIC finufft  dfs fftw3 fftw3_threads "-Wl,-no_compact_unwind")

add_test(dfs_surfgrad_test dfs_surfgrad_test)

# Test double dot product
#add_executable(dfs_dotgrad_test dfs_dotgrad_test.cpp)
#target_include_directories(dfs_dotgrad_test PUBLIC ${PROJECT_SOURCE_DIR}/src ${FFTW3_INCLUDE_DIR} ${FFTW3THREADS_INCLUDE_DIR}
#)
#target_link_libraries(dfs_dotgrad_test PUBLIC finufft  dfs fftw3 fftw3_threads "-Wl,-no_compact_unwind")

#add_test(dfs_dotgrad_test dfs_dotgrad_test)





















