#!/bin/bash

rm -rf CMake*

if [ $1 = "fresh" ] ; then
    rm -rf Testing/
    rm -rf tpl/
    rm -rf CTest*
    rm -rf Dart*
    rm -rf Makefile *.cmake
    rm -rf Compadre-prefix/
    rm -rf Kokkos-prefix/
    rm Lpm*h
    rm -rf src/
    rm -rf tests/
fi

if [ "$HOSTNAME" = "s979562" ] || [ "$HOSTNAME" = "s979598" ] ; then
    echo "hostname = $HOSTNAME"
    SRC_ROOT=$HOME/lpm
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
-D CMAKE_VERBOSE_MAKEFILE:BOOL=ON \
$EXTRA_ARGS \
$SRC_ROOT
