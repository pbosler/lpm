#!/bin/bash

module purge
module load cmake/3.19.3
module load git
module load gcc/7.2.0
module load openmpi/2.1.2/gcc/7.2.0
module load openblas/0.2.20/gcc/7.2.0
module unload python
module load python/3.7.3
module load hdf5/1.10.1/openmpi/2.1.2/gcc/7.2.0
module load netcdf/4.4.1.1/openmpi/2.1.2/gcc/7.2.0
module load yamlcpp

TR=/ascldap/users/pabosle/trilinos-lpm
#TR=/ascldap/users/pabosle/trilinos-lpm-debug

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

export YAMLCPP_INCLUDE_DIR=$YAMLCPP_ROOT/include
export YAMLCPP_LIBRARY_DIR=$YAMLCPP_ROOT/lib
export YAMLCPP_LIBRARY=libyaml-cpp.a
