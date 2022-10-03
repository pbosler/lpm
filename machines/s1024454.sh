#!/bin/bash

# if [[ $OMPI_CXX == *"nvcc_wrapper"* ]]; then
#   export TR=
# else
#   export TR=/home/pabosle/trilinos-lpm-omp
# fi
export Trilinos_DIR=/home/pabosle/trilinos-lpm-omp

export HDF5_INCLUDE_DIR=$HDF5_ROOT/include
export HDF5_LIBRARY_DIR=$HDF5_ROOT/lib
export HDF5_LIBRARY=libhdf5.so
export HDF5_HL_LIBRARY=libhdf5_hl.so

export NETCDF_INCLUDE_DIR=$NETCDF_C_ROOT/include
export NETCDF_LIBRARY_DIR=$NETCDF_C_ROOT/lib
export NETCDF_LIBRARY=libnetcdf.so

# export VTK_INCLUDE_DIR=$VTK_ROOT/include/vtk-8.2
# export VTK_LIBRARY_DIR=$VTK_ROOT/lib64
#
# export BOOST_INCLUDE_DIR=$BOOST_ROOT/include
# export BOOST_LIBRARY_DIR=$BOOST_ROOT/lib

export YAMLCPP_INCLUDE_DIR=$YAML_CPP_ROOT/include
export YAMLCPP_LIBRARY_DIR=$YAML_CPP_ROOT/lib
export YAMLCPP_LIBRARY=libyaml-cpp.so

export SPDLOG_INCLUDE_DIR=$SPDLOG_ROOT/include
export SPDLOG_LIBRARY_DIR=$SPDLOG_ROOT/lib64
export SPDLOG_LIBRARY=libspdlog.a
