#!/bin/bash

FINUFFT_SRC=$HOME/finufft
FINUFFT_BUILD_DIR=$FINUFFT_SRC/build-openmp
LPM_TPL_DIR=$HOME/lpm-tpl-debug
FFTW_LIB=$HOME/lpm-tpl-debug/lib64/libfftw3.a

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
rm -rf $FINUFFT_BUILD_DIR/CMake*

cmake -Wno-dev -B $FINUFFT_BUILD_DIR \
-DCMAKE_CXX_STANDARD=17 \
-DCMAKE_BUILD_TYPE=Debug \
-DCMAKE_CXX_COMPILER=mpicxx \
-DCMAKE_INSTALL_PREFIX=$LPM_TPL_DIR \
-DCMAKE_C_FLAGS="-I${LPM_TPL_DIR}/include" \
-DCMAKE_CXX_FLAGS="-I${LPM_TPL_DIR}/include" \
-DFINUFFT_FFTW_LIBRARIES=$FFTW_LIB \
-DFINUFFT_USE_OPENMP=ON \
-DFINUFFT_ENABLE_SANITIZERS=OFF \
-DFINUFFT_BUILD_TESTS=OFF \
-DFINUFFT_USE_CPU=ON \
-DFINUFFT_USE_CUDA=OFF \
$FINUFFT_SRC
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