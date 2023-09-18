#!/bin/bash

# start fresh with CMake
rm -rf CMake* *.cmake

# Kokkos and KokkosKernels must exist already
KOKKOS_INSTALL=$HOME/kokkos-debug/install
KOKKOS_KERNELS_INSTALL=$KOKKOS_INSTALL

# get the source: git clone git@github.com:sandialabs/compadre.git
COMPADRE_SRC=$HOME/compadre

cmake \
-DCMAKE_BUILD_TYPE=DEBUG \
-DCMAKE_INSTALL_PREFIX=$HOME/compadre-debug \
-DKokkosCore_PREFIX=$KOKKOS_INSTALL \
-DKokkosKernels_PREFIX=$KOKKOS_KERNELS_INSTALL \
$COMPADRE_SRC
