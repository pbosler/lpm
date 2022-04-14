#!/bin/bash

export MPI_ROOT=$HOME/openmpi-3.1.6/gcc-8.5
export BOOST_ROOT=$HOME/boost-1.77.0
export HDF5_ROOT=$HOME/hdf5-1.10.8/gcc-8.5
export NETCDF_ROOT=$HOME/netcdf-4.8.1/gcc-8.5
export VTK_ROOT=$HOME/vtk-8.2.0/gcc-8.5


if [[ $OMPI_CXX == *"nvcc_wrapper"* ]]; then
  TR=/home/pabosle/trilinos-lpm-cuda
else
  TR=/home/pabosle/trilinos-lpm-omp
fi

export HDF5_INCLUDE_DIR=$HDF5_ROOT/include
export HDF5_LIBRARY_DIR=$HDF5_ROOT/lib
export HDF5_LIBRARY=libhdf5.a
export HDF5_HL_LIBRARY=libhdf5_hl.a

export NETCDF_INCLUDE_DIR=$NETCDF_ROOT/include
export NETCDF_LIBRARY_DIR=$NETCDF_ROOT/lib
export NETCDF_LIBRARY=libnetcdf.a

export VTK_INCLUDE_DIR=$VTK_ROOT/include/vtk-8.2
export VTK_LIBRARY_DIR=$VTK_ROOT/lib64

export BOOST_INCLUDE_DIR=$BOOST_ROOT/include
export BOOST_LIBRARY_DIR=$BOOST_ROOT/lib

export TRILINOS_INCLUDE_DIR=$TR/include
export TRILINOS_LIBRARY_DIR=$TR/lib
export TRILINOS_TPL_INCLUDE_DIR=$TR/include
export TRILINOS_TPL_LIBRARY_DIR=$TR/lib

export YAMLCPP_INCLUDE_DIR=$YAMLCPP_ROOT/include
export YAMLCPP_LIBRARY_DIR=$YAMLCPP_ROOT/lib64
export YAMLCPP_LIBRARY=libyaml-cpp.a

export SPDLOG_INCLUDE_DIR=$SPDLOG_ROOT/include
export SPDLOG_LIBRARY_DIR=$SPDLOG_ROOT/lib64
export SPDLOG_LIBRARY=libspdlog.a
