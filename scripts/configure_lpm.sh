
# start with a fresh CMake slate
rm -rf CMake* *.cmake

LPM_SRC=$HOME/lpm

# define your TPL paths (these must exist already)
export COMPOSE=$HOME/compose/build
export KOKKOS=$HOME/kokkos-debug/install
export COMPADRE=$HOME/compadre-debug

cmake \
-DCMAKE_INSTALL_PREFIX=./install \
-DCMAKE_BUILD_TYPE=Debug \
-DCMAKE_CXX_COMPILER=g++ \
-DCMAKE_C_COMPILER=gcc \
-DLPM_PRECISION=double \
-DKokkos_DIR=$KOKKOS \
-DKokkosKernels_DIR=$KOKKOS \
-DCompadre_DIR=$COMPADRE \
-DLPM_ENABLE_Compose=ON \
-DCompose_DIR=$COMPOSE \
-DLPM_ENABLE_VTK=ON \
-G"Unix Makefiles" \
$LPM_SRC
