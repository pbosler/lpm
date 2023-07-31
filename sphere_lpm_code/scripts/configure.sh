#!/bin/bash
#message(STATUS "set env var")
export CC="/opt/homebrew/opt/llvm/bin/clang"
export CXX="/opt/homebrew/opt/llvm/bin/clang++"
export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"

rm -rf CMake*

cmake \
-D Kokkos_DIR=$HOME/kokkos-debug/install \
-D KokkosKernels_DIR=$HOME/kokkos-debug/install \
-D FFTW3_DIR=/opt/homebrew \
-D FFTW3THREADS_DIR=/opt/homebrew \
.. 

