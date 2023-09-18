#!/bin/bash

# start fresh with CMake
rm -rf CMake* *.cmake

# get the source (Release 3.7.02)
#   git clone git@github.com:kokkos/kokkos.git
#   cd kokkos
#   git checkout tags/3.7.02
KOKKOS_SRC=$HOME/kokkos

cmake \
-D CMAKE_BUILD_TYPE=Debug \
-D CMAKE_CXX_STANDARD=17 \
-D CMAKE_INSTALL_PREFIX=./install \
-D Kokkos_ENABLE_SERIAL=ON \
-D Kokkos_ENABLE_THREADS=ON \
-D Kokkos_ENABLE_LIBDL=ON \
$KOKKOS_SRC
