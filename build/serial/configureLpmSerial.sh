#!/bin/bash

rm -rf CMakeFiles/ CMakeCache.txt

if [ "$HOSTNAME" = "s979598" ] ; then
    echo "hostname = $HOSTNAME"
    SRC_ROOT=$HOME/repos/lpmkokkos
    export VTK_ROOT=$HOME/installs/vtk-8.1.1
    export KO=$HOME/kokkos-serial-debug
    export CPTK=$HOME/repos/cp_toolkit/build
elif [ "$HOSTNAME" = "s979562" ] ; then
    echo "hostname = $HOSTNAME"
    SRC_ROOT=$HOME/lpm
    export VTK_ROOT=$HOME/installs/vtk-8.1.1
    export KO=$HOME/kokkos-serial-debug
    export CPTK=$HOME/compadre-serial-debug
else
    SRC_ROOT=$HOME/lpmkokkos
    export VTK_ROOT=$HOME/VTK-8.1.1
    export KO=$HOME/kokkos-serial
    export CPTK=$HOME/compadre/build
    export SP=$HOME/spherepack3.2
fi

EXTRA_ARGS=$1
export OMPI_CXX=g++
# 


cmake -Wno-dev \
-D CMAKE_BUILD_TYPE:STRING="DEBUG" \
-D CMAKE_CXX_FLAGS="-g" \
-D Kokkos_ROOT=$KO \
-D Compadre_ROOT=$CPTK \
-D Spherepack_ROOT=$SP \
-D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
$EXTRA_ARGS \
$SRC_ROOT
