#!/bin/bash

module purge
module load sems-env
module load sems-doxygen
module load sems-cmake
module load sems-graphviz
module load sems-git/2.10.1
module load sems-tex/2019
module load sems-zlib/1.2.8/base

module load sems-gdb/7.9.1
module load sems-openmpi/4.0.5
module load sems-cuda/10.1
module load sems-gcc/8.3.0
module load sems-boost/1.70.0/base

#module load sems-clang/10.0.0 # for clang-format
export HDF5_ROOT=$HOME/hdf5-1.10.2/gcc-7.3.1
export PATH=$PATH:$HDF5_ROOT/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HDF5_ROOT/lib

export NETCDF_ROOT=/ascldap/users/pabosle/netcdf-4.6.1/gcc-7.3.1
export PATH=$PATH:$NETCDF_ROOT/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NETCDF_ROOT/lib

export BOOST_ROOT=$SEMS_BOOST_ROOT

export VTK_ROOT=/ascldap/users/pabosle/vtk-8.1.2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VTK_ROOT/lib

export FC=mpifort
export CC=mpicc
export CXX=mpicxx

alias mpiomp="export OMPI_CXX=g++"
alias mpicuda="export OMPI_CXX=/ascldap/users/pabosle/kokkos/bin/nvcc_wrapper"

if [[ $OMPI_CXX == *"nvcc_wrapper"* ]]; then
  TR=/ascldap/users/pabosle/trilinos-lpm-gpu
else
  TR=/ascldap/users/pabosle/trilinos-lpm-cpu
  #TR=/ascldap/users/pabosle/trilinos-lpm-cpu-debug
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

export BOOST_INCLUDE_DIR=$BOOST_ROOT/include
export BOOST_LIBRARY_DIR=$BOOST_ROOT/lib

export TRILINOS_INCLUDE_DIR=$TR/include
export TRILINOS_LIBRARY_DIR=$TR/lib
export TRILINOS_TPL_INCLUDE_DIR=$TR/include
export TRILINOS_TPL_LIBRARY_DIR=$TR/lib
