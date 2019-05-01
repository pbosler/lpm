#!/bin/bash

rm -rf CMakeFiles/ CMakeCache.txt

if [ "$HOSTNAME" = "s979562" ] || [ "$HOSTNAME" -eq "s979598" ] ; then
    echo "hostname = $HOSTNAME"
    SRC_ROOT=$HOME/repos/lpmkokkos
    export VTK_ROOT=$HOME/installs/vtk-8.1.1
else
    SRC_ROOT=$HOME/lpmkokkos
    export VTK_ROOT=$HOME/VTK-8.1.1
fi

EXTRA_ARGS=$1
#export KO=$HOME/installs/kokkos-serial-debug
export KO=$HOME/kokkos-serial
export OMPI_CXX=g++
# 


cmake -Wno-dev \
-D CMAKE_BUILD_TYPE:STRING="DEBUG" \
-D CMAKE_CXX_FLAGS="-g" \
-D Kokkos_ROOT=$KO \
-D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
$EXTRA_ARGS \
$SRC_ROOT
