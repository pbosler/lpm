#!/bin/bash

LPM_SRC=$HOME/lpm
BLD_DIR=$HOME/lpm/build-openmp

rm -rf $BLD_DIR/CMake* $BLD_DIR/*.cmake

export TPL_DIR=$HOME/lpm-tpl-openmp
export SPDLOG_DIR=$HOME/lpm-tpl-cuda
export CATCH2_DIR=$HOME/lpm-tpl-cuda
export VTK_DIR=$HOME/lpm-tpl-cuda

cmake -B $BLD_DIR \
-DCMAKE_CXX_COMPILER=mpic++ \
-DCMAKE_CXX_STANDARD=17 \
-DCMAKE_BUILD_TYPE=RelWithDebInfo \
-DCMAKE_Fortran_COMPILER=mpifort \
-DCMAKE_C_COMPILER=mpicc \
-DLPM_ENABLE_Compose=ON \
-DLPM_ENABLE_DFS=ON \
-DLPM_TPL_DIR=$TPL_DIR \
-DKokkos_DIR=$TPL_DIR \
-DKokkosKernels_DIR=$TPL_DIR \
-DCompadre_DIR=$TPL_DIR \
-DCompose_DIR=$TPL_DIR \
-DSPDLOG_DIR=$SPDLOG_DIR \
-DCATCH2_DIR=$CATCH2_DIR \
-DVTK_DIR=$VTK_DIR \
-DFFTW3_DIR=$TPL_DIR \
-DFINUFFT_DIR=$TPL_DIR \
-G"Unix Makefiles" \
$LPM_SRC

