#!/bin/bash

rm -rf CMakeCache.txt CMakeFiles/

TRILINOS_SRC=$HOME/Trilinos
# export CUDA_LAUNCH_BLOCKING=1

BUILD_DIR=`pwd`

cmake \
-DCMAKE_BUILD_TYPE:STRING=DEBUG \
-DTrilinos_ENABLE_Fortran:BOOL=OFF \
-DCMAKE_INSTALL_PREFIX:PATH=${BUILD_DIR}/install \
\
-DTPL_ENABLE_MPI:BOOL=ON \
-DTrilinos_ENABLE_OpenMP:BOOL=OFF \
-DTPL_ENABLE_Pthread:BOOL=OFF \
\
-DTPL_ENABLE_Boost:BOOL=ON \
-DBoost_INCLUDE_DIRS:FILEPATH=$BOOST_ROOT/include \
-DBoost_LIBRARY_DIRS:FILEPATH=$BOOST_ROOT/lib \
\
-DTPL_ENABLE_Netcdf:BOOL=ON\
-DNetcdf_INCLUDE_DIRS:FILEPATH=$NETCDF_ROOT/include \
-DNetcdf_LIBRARY_DIRS:FILEPATH=$NETCDF_ROOT/lib \
-DTPL_Netcdf_INCLUDE_DIRS=$NETCDF_ROOT/include \
\
-DTPL_ENABLE_CUDA:BOOL=ON \
-DCMAKE_CXX_FLAGS="-g -lineinfo -Xcudafe \
--diag_suppress=conversion_function_not_usable -Xcudafe \
--diag_suppress=cc_clobber_ignored -Xcudafe \
--diag_suppress=code_is_unreachable" \
-DTPL_ENABLE_CUBLAS=ON \
-DTPL_ENABLE_CUSOLVER=ON \
\
-DTrilinos_ENABLE_ALL_PACKAGES:BOOL=OFF \
-DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF \
\
-DTrilinos_ENABLE_Teuchos:BOOL=ON \
\
-DTrilinos_ENABLE_Kokkos:BOOL=ON \
-DTrilinos_ENABLE_KokkosCore:BOOL=ON \
-DTrilinos_ENABLE_KokkosAlgorithms:BOOL=ON \
-DTrilinos_ENABLE_KokkosKernels:BOOL=ON \
-DKokkos_ENABLE_SERIAL:BOOL=ON \
-DKokkos_ENABLE_OPENMP:BOOL=OFF \
-DKokkos_ENABLE_PTHREAD:BOOL=OFF \
-DKokkos_ENABLE_CUDA:BOOL=ON \
-DKokkos_ARCH_MAXWELL52=ON \
\
-DTrilinos_ENABLE_Compadre:BOOL=ON \
\
-DTrilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON \
\
$TRILINOS_SRC
