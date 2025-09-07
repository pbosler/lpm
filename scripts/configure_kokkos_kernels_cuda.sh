#!/bin/bash

KOKKOS_KERNELS_SRC=$HOME/kokkos-kernels
KOKKOS_KERNELS_BUILD_DIR=$HOME/kokkos-kernels/build-cuda
LPM_TPL_DIR=$HOME/lpm-tpl-cuda

configFlag=
buildFlag=
installFlag=
testFlag=

while getopts 'cbti' OPTION
do
  case $OPTION in
    c) configFlag=1
       ;; 
    b) buildFlag=1
       ;;
    t) testFlag=1
       ;;
    i) installFlag=1
       ;;
    ?) printf "Usage: %s: [-cbti] args\n" $(basename $0) >&2
       exit 2
       ;;
  esac
done
shift $(($OPTIND -1))


if [ "$configFlag" ]
then
rm -rf $KOKKOS_KERNELS_BUILD_DIR/CMake*

cmake -B $KOKKOS_KERNELS_BUILD_DIR \
-DCMAKE_CXX_COMPILER=mpicxx \
-DCMAKE_INSTALL_PREFIX=$LPM_TPL_DIR \
-DKokkos_ROOT=$LPM_TPL_DIR \
$KOKKOS_KERNELS_SRC
fi

if [ "$buildFlag" ]
then
cmake --build $KOKKOS_KERNELS_BUILD_DIR -j 16
fi

if [ "$testFlag" ]
then
ctest --test-dir $KOKKOS_KERNELS_BUILD_DIR --output-on-failure
fi

if [ "$installFlag" ]
then
cmake --install $KOKKOS_KERNELS_BUILD_DIR --prefix $LPM_TPL_DIR
fi