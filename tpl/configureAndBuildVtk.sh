#!/bin/bash

#
#   NOTE: This script may require several minutes to complete.
#
#   Download a VTK tarball and extract it into $VTK_SRC.
#     change the variables below:
#           VTK_SRC=/path/to/extracted/tarball
#           VTK_DEST=/path/to/desired/install/location
#
#   Make a VTK build directory, e.g.,
#       mkdir $VTK_SRC/build
#   
#   Run this script from the VTK build directory.
#       cd $VTK_SRC/build
#       cp $LPM_ROOT/scripts/configureAndBuildVtk.sh .
#
#       rm -rf CMake*  (recommended)
#
#       ./configureAndBuildVtk.sh
#

VTK_SRC=$HOME/tarballs/VTK-8.1.0
VTK_DEST=$HOME/testbuild/vtk-8.1.0

cmake \
-D CMAKE_BUILD_TYPE="RelWithDebInfo" \
-D CMAKE_INSTALL_PREFIX=$VTK_DEST \
-D VTK_Group_MPI=ON \
-D VTK_Group_Rendering=ON \
-D VTK_Group_StandAlone=ON \
$VTK_SRC

make -j 12 && make install

