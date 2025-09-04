#!/bin/bash

LPM_SRC=$HOME/lpm
BLD_DIR=$HOME/lpm/build-cuda

rm -rf $BLD_DIR/CMake* $BLD_DIR/*.cmake

export TPL_DIR=$HOME/lpm-tpl-cuda
export OMPI_CXX=$HOME/kokkos/bin/nvcc_wrapper

cmake -B $BLD_DIR \
-DCMAKE_CXX_COMPILER=mpic++ \
-DCMAKE_CXX_STANDARD=17 \
-DCMAKE_BUILD_TYPE=RelWithDebInfo \
-DCMAKE_Fortran_COMPILER=mpifort \
-DCMAKE_C_COMPILER=mpicc \
-DLPM_ENABLE_Compose=ON \
-DLPM_ENABLE_DFS=OFF \
-DLPM_TPL_DIR=$TPL_DIR \
-DKokkos_DIR=$TPL_DIR \
-DKokkosKernels_DIR=$TPL_DIR \
-DCompadre_DIR=$TPL_DIR \
-DCompose_DIR=$TPL_DIR \
-G"Unix Makefiles" \
$LPM_SRC

