#!/bin/bash

rm -rf CMakeFiles/ CMakeCache.txt

# ** BEFORE ** you run this script
# 1. Change SRC_ROOT to point to your own clone of lpmkokkos
#

SRC_ROOT=$HOME/lpm


EXTRA_ARGS=$1
export TR=/ascldap/users/pabosle/trilinos-cuda/install
export SP=/ascldap/users/pabosle/spherepack3.2

export OMPI_CXX=$TR/bin/nvcc_wrapper

cmake -Wno-dev \
-D CMAKE_BUILD_TYPE:STRING=RELEASE \
-D Trilinos_ROOT=$TR \
-D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
-D Spherepack_ROOT=$SP \
$EXTRA_ARGS \
$SRC_ROOT
