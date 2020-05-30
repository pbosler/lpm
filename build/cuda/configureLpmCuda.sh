#!/bin/bash

rm -rf CMakeFiles/ CMakeCache.txt

SRC_ROOT=$HOME/aj/lpmkokkos
EXTRA_ARGS=$1
export TR=$HOME/aj/trilinos-cuda-debug/install
export SP=$HOME/spherepack3.2

cmake -Wno-dev \
-D CMAKE_BUILD_TYPE:STRING="DEBUG" \
-D Trilinos_ROOT=$TR \
-D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
-D Spherepack_ROOT=$SP \
$EXTRA_ARGS \
$SRC_ROOT
