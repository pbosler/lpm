#!/bin/bash

KOKKOS_SRC=$HOME/kokkos
KOKKOS_BUILD_DIR=$HOME/kokkos/build-openmp
LPM_TPL_DIR=$HOME/lpm-tpl-openmp

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
    ?) printf "Usage: %s: [-cbr] args\n" $(basename $0) >&2
       exit 2
       ;;
  esac
done
shift $(($OPTIND -1))


if [ "$configFlag" ]
then
rm -rf $KOKKOS_BUILD_DIR/CMake*

cmake -B $KOKKOS_BUILD_DIR \
-DCMAKE_CXX_COMPILER=mpic++ \
-DCMAKE_CXX_STANDARD=17 \
-DCMAKE_CXX_FLAGS="-fPIC" \
-DCMAKE_BUILD_TYPE=RelWithDebInfo \
-DKokkos_ENABLE_SERIAL=ON \
-DKokkos_ENABLE_OPENMP=ON \
-DKokkos_ENABLE_CUDA=OFF \
-DKokkos_ARCH_NATIVE=ON \
-DKokkos_ENABLE_DEPRECATED_CODE_4=ON \
-DKokkos_ENABLE_DEPRECATION_WARNINGS=ON \
-DKokkos_ENABLE_TESTS=OFF \
$KOKKOS_SRC
fi

if [ "$buildFlag" ]
then
cmake --build $KOKKOS_BUILD_DIR -j 24
fi

if [ "$testFlag" ]
then
ctest --test-dir $KOKKOS_BUILD_DIR --output-on-failure
fi

if [ "$installFlag" ]
then
cmake --install $KOKKOS_BUILD_DIR --prefix $LPM_TPL_DIR
fi