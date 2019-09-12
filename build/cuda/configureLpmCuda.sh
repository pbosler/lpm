#!/bin/bash

rm -rf CMakeFiles/ CMakeCache.txt

SRC_ROOT=$HOME/lpm
EXTRA_ARGS=$1
export KO=$HOME/kokkos-cuda/gcc-6.3.1
export CP=$HOME/compadre/build/cuda
export SP=$HOME/spherepack3.2
export OMP_CXX=$HOME/kokkos/bin/nvcc_wrapper

cmake -Wno-dev \
-D CMAKE_BUILD_TYPE:STRING="RelWithDebInfo" \
-D Kokkos_ROOT=$KO \
-D Compadre_ROOT=$CP \
-D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
-D Spherepack_ROOT=$SP \
$EXTRA_ARGS \
$SRC_ROOT
