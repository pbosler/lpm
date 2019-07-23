#!/bin/bash

rm -rf CMakeFiles/ CMakeCache.txt

SRC_ROOT=$HOME/lpm
EXTRA_ARGS=$1
export KO=$HOME/kokkos-cuda/gcc-6.3.1
export CP=$HOME/compadre/build/cuda

cmake -Wno-dev \
-D CMAKE_BUILD_TYPE:STRING="DEBUG" \
-D Kokkos_ROOT=$KO \
-D Compadre_ROOT=$CP \
-D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
$EXTRA_ARGS \
$SRC_ROOT
