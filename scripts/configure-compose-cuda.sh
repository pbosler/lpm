#!/bin/bash

COMPOSE_SRC=$HOME/compose
COMPOSE_BUILD_DIR=$HOME/compose/build-cuda
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
rm -rf $COMPOSE_BUILD_DIR/CMake*

cmake -B $COMPOSE_BUILD_DIR \
-DCMAKE_CXX_STANDARD=17 \
-DCMAKE_BUILD_TYPE=RelWithDebInfo \
-DCMAKE_CXX_FLAGS="-std=c++17 -fPIC" \
-DCMAKE_CXX_COMPILER=mpicxx \
-DCMAKE_INSTALL_PREFIX=$LPM_TPL_DIR \
-DKokkos_DIR=$LPM_TPL_DIR 
fi

if [ "$buildFlag" ]
then
cmake --build $COMPOSE_BUILD_DIR -j 16
fi

if [ "$testFlag" ]
then
ctest --test-dir $COMPOSE_BUILD_DIR --output-on-failure
fi

if [ "$installFlag" ]
then
cmake --install $COMPOSE_BUILD_DIR --prefix $LPM_TPL_DIR
fi