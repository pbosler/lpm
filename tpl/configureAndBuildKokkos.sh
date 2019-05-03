#!/bin/bash

#
#   NOTE: This script should run quickly (< 3 minutes).
#
#   Get the kokkos src code:
#       From your home directory, run:
#       git clone https://github.com/kokkos/kokkos.git
#
#       *for other locations, change the Kokkos_SRC variable below.
#       
#
#   Make a build directory (this will also be the install directory); e.g.,
#       mkdir kokkos-openmp
#
#       *for other install locations, change the Kokkos_DST variable below.
#
#   Copy this script to the build directory
#       cd kokkos-openmp
#       cp lpmkokkos/tpl/configureAndBuildKokkos.sh .
#       
#       rm -rf CMake*  (recommended)
#
#   Run it:
#       ./configureAndBuildKokkos.sh
#
#

Kokkos_SRC=$HOME/kokkos
Kokkos_DST=`pwd`

# If a Cuda build, add the line
#
#   -D CMAKE_CXX_COMPILER=$Kokkos_SRC/bin/nvcc_wrapper
#

cmake \
-D KOKKOS_ENABLE_DEBUG=FALSE \
-D KOKKOS_ENABLE_OPENMP=TRUE \
-D KOKKOS_ENABLE_CUDA=FALSE \
-D KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION=FALSE \
-D KOKKOS_ENABLE_DEPRECATED_CODE=FALSE \
-D KOKKOS_ENABLE_EXPLICIT_INSTANTIATION=FALSE \
-D CMAKE_INSTALL_PREFIX=$Kokkos_DST \
$Kokkos_SRC

make -j 8 clean
make -j 8 && make install
