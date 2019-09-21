#!/bin/bash

rm -rf CMakeFiles/ CMakeCache.txt

SRC_ROOT=$HOME/lpm
EXTRA_ARGS=$1
if [ "$HOSTNAME" = "s1046231" ]; then
	export KO=$HOME/kokkos-openmp
else
	export KO=$HOME/kokkos-openmp/gcc-6.3.1
fi
export CP=$HOME/compadre/build/openmp
export SP=$HOME/spherepack3.2
export OMPI_CXX=g++

cmake \
-D CMAKE_BUILD_TYPE:STRING="RelWithDebInfo" \
-D Kokkos_ROOT=$KO \
-D Compadre_ROOT=$CP \
-D Spherepack_ROOT=$SP \
-D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
$EXTRA_ARGS \
$SRC_ROOT
