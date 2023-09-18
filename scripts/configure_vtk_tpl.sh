#!/bin/bash

# start fresh with CMake
rm -rf CMake*

# get the source:
#   Download the VTK release 9.2.2 tarball from vtk.org and
#   unpack it.
VTK_SRC=$HOME/tarballs/VTK-9.2.2

cmake \
  -DCMAKE_INSTALL_PREFIX=$HOME/vtk \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DVTK_USE_CUDA=OFF \
  -DVTK_USE_MPI=ON \
  -DVTK_WRAP_PYTHON=OFF \
  -DVTK_USE_EXTERNAL=OFF \
  -GNinja \
  $VTK_SRC
