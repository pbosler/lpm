#!/bin/bash

# start fresh with CMake
rm -rf CMake*

# Kokkos must have been installed already
KOKKOS_INSTALL_DIR=$HOME/kokkos-debug/install

# get the source (release 3.7.02)
#   git clone git@github.com:kokkos/kokkos-kernels.git
#   cd kokkos-kernels
#   git checkout tags/3.7.02
KOKKOS_KERNELS_SRC=$HOME/kokkos-kernels

cmake \
-DCMAKE_INSTALL_PREFIX=$KOKKOS_INSTALL_DIR \
-DCMAKE_CXX_STANDARD=17 \
-DCMAKE_BUILD_TYPE=DEBUG \
-DKokkos_ROOT=$KOKKOS_INSTALL_DIR \
$KOKKOS_KERNELS_SRC
