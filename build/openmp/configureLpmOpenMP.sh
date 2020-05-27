#!/bin/bash

rm -rf CMakeFiles/ CMakeCache.txt

SRC_ROOT=$HOME/aj/lpmkokkos
EXTRA_ARGS=$1
if [ "$HOSTNAME" = "s1046231" ]; then
	export KO=$HOME/kokkos-openmp
else
	export KO=$HOME/aj/kokkos-openmp/install
fi
#export CP=$HOME/aj/compadre/build/openmp/install
export SP=$HOME/spherepack3.2
export OMPI_CXX=g++
export TR=$HOME/aj/trilinos-openmp-debug/install

cmake \
-D CMAKE_BUILD_TYPE:STRING="DEBUG" \
-D CMAKE_CXX_FLAGS="-fopenmp -fPIC" \
-D LPM_ENABLE_DEBUG=ON \
-D Trilinos_ROOT=$TR \
-D Spherepack_ROOT=$SP \
-D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
$EXTRA_ARGS \
$SRC_ROOT
