#!/bin/bash

rm -rf CMakeFiles/ CMakeCache.txt

SRC_ROOT=$HOME/repos/lpmkokkos
EXTRA_ARGS=$1
export KO=$HOME/installs/kokkos-serial-debug
export OMPI_CXX=g++
export VTK_ROOT=$HOME/installs/vtk-8.1.1

cmake -Wno-dev \
-D CMAKE_BUILD_TYPE:STRING="DEBUG" \
-D CMAKE_CXX_FLAGS="-g" \
-D Kokkos_ROOT=$KO \
-D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
$EXTRA_ARGS \
$SRC_ROOT
