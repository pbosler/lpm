#!/bin/bash

# start fresh with CMake
rm -rf CMake* *.cmake

# Kokkos must have been installed already
KOKKOS=$HOME/kokkos-debug/install

# get the source (branch lpm-tpl):
#   git clone git@github.com:pbosler/COMPOSE.git compose
#   cd compose
#   git checkout lpm-tpl
COMPOSE_SRC=$HOME/compose

cmake \
-DCMAKE_CXX_STANDARD=17 \
-DBUILD_SHARED_LIBS=OFF \
-DCMAKE_BUILD_TYPE=DEBUG \
-DKokkos_DIR=$KOKKOS \
-DCMAKE_INSTALL_PREFIX=../install-debug \
$COMPOSE_SRC
