#
# # Print usage info.
# if [ "$1" = "" ]; then
#   echo "setup: Creates a build directory with a configuration file."
#   echo "Usage: configure_lpm.sh <build_dir>"
#   exit 1
# fi
#
# # Create the build directory if it doesn't exist.
# if [ ! -d $1 ]; then
#   mkdir -p $1
# fi
#
# cd $1
rm -rf CMake*

LPM_SRC=$HOME/lpm
export Trilinos_DIR=$HOME/trilinos-lpm/lib/cmake/Trilinos

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
-DLPM_ENABLE_CUDA=OFF \
-DLPM_ENABLE_VTK=ON \
-DLPM_ENABLE_BOOST=OFF \
-DLPM_ENABLE_NETCDF=OFF \
-DHDF5_INCLUDE_DIR=$HDF5_INCLUDE_DIR \
-DHDF5_LIBRARY=$HDF5_LIBRARY_DIR/$HDF5_LIBRARY \
-DHDF5_HL_LIBRARY=$HDF5_LIBRARY_DIR/$HDF5_HL_LIBRARY \
-DNETCDF_INCLUDE_DIR=$NETCDF_INCLUDE_DIR \
-DNETCDF_LIBRARY=$NETCDF_LIBRARY_DIR/$NETCDF_LIBRARY \
-G"Unix Makefiles" \
$LPM_SRC
