#!/bin/bash

if [[ $OMPI_CXX == *"nvcc_wrapper"* ]]; then
  TR=$HOME/trilinos-lpm-gpu
else
  TR=$HOME/trilinos-lpm-cpu
  #TR=$HOME/trilinos-lpm-cpu-debug
fi

export HDF5_INCLUDE_DIR=$HDF5_ROOT/include
export HDF5_LIBRARY_DIR=$HDF5_ROOT/lib
export HDF5_LIBRARY=libhdf5.a
export HDF5_HL_LIBRARY=libhdf5_hl.a

export NETCDF_INCLUDE_DIR=$NETCDF_ROOT/include
export NETCDF_LIBRARY_DIR=$NETCDF_ROOT/lib
export NETCDF_LIBRARY=libnetcdf.a

export VTK_INCLUDE_DIR=$VTK_ROOT/include/vtk-8.1
export VTK_LIBRARY_DIR=$VTK_ROOT/lib

export TRILINOS_INCLUDE_DIR=$TR/include
export TRILINOS_LIBRARY_DIR=$TR/lib
export TRILINOS_TPL_INCLUDE_DIR=$TR/include
export TRILINOS_TPL_LIBRARY_DIR=$TR/lib
