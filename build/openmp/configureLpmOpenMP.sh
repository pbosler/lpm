#!/bin/bash

rm -rf CMakeFiles/ CMakeCache.txt
#
# **BEFORE** running this script, set SRC_ROOT to your local clone of lpmkokkos
#
SRC_ROOT=$HOME/lpm
EXTRA_ARGS=$1
if [ "$HOSTNAME" = "s1046231" ]; then
  echo $HOSTNAME
	export TR=$HOME/trilinos-openmp-debug/install
elif [ "$HOSTNAME" = "s1024454" ]; then
  echo $HOSTNAME
  export TR=/ascldap/users/pabosle/trilinos-openmp-debug/install
else
  export TR=$HOME/trilinos-openmp
fi
#export CP=$HOME/aj/compadre/build/openmp/install
export SP=$HOME/spherepack3.2
export OMPI_CXX=g++

cmake \
-D CMAKE_BUILD_TYPE:STRING="DEBUG" \
-D CMAKE_CXX_FLAGS="-fopenmp -fPIC" \
-D LPM_ENABLE_DEBUG=ON \
-D Trilinos_ROOT=$TR \
-D Spherepack_ROOT=$SP \
-D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
$EXTRA_ARGS \
$SRC_ROOT
