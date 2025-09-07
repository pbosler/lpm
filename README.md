# LPM: Lagrangian Particle methods

![Auto-test](https://github.com/pbosler/lpm/actions/workflows/auto_test.yml/badge.svg)

LPM contains several algorithms for computing solutions to PDEs using Lagrangian Particle Methods.  Generally, these methods are "mesh free," in the sense that the numerical methods used to solve the PDE and advance the simulation forward in time do not require a mesh topology to exist between particles.  Optionally, we support meshes that are defined recursively in a quad-tree structure (for 2D problems) for use with adaptive refinement.

## Dependencies

LPM has a few dependencies on third-party libraries (TPLs):
- [Catch2](https://github.com/catchorg/Catch2): Unit testing framework
- [Spdlog](https://github.com/gabime/spdlog): Thread-safe logging to console
- [Kokkos](https://github.com/kokkos/kokkos): Kokkos performance portability library
- [KokkosKernels](https://github.com/kokkos/kokkos-kernels): Kokkos Kernels batched linear algebra
- [Compadre](https://github.com/sandialabs/compadre): Meshfree approximations using local stencils, for both Cartesian space and non-Euclidean manifolds
- [COMPOSE](https://github.com/E3SM-Project/COMPOSE): Semi-Lagrangian methods specialized for the sphere, plus general property preservation functions
- [VTK](vtk.org): For data saving and visualization with either VTK or [Paraview](paraview.org).

To optionally enable the double Fourier series fast solver for spherical problems, some additional libraries are necessary:
- [FFTW3](fftw3.org) Fast Fourier Transforms (FFT)
- [FINUFFT](https://github.com/flatironinstitute/finufft) Non-uniform FFT allows for basis expansions at locations that are not FFT grid points

Example scripts to build each of these libraries may be found in the [scripts folder](https://github.com/pbosler/lpm/tree/main/scripts). These scripts are neither maintained nor tested in the automated workflow, so they may become out of date.  The specific commands used to build the TPLs for automated testing may be found in the [tools folder](https://github.com/pbosler/lpm/tree/common-tpls/tools)' Dockerfile, which also contains the exact versions of each library used by our testing workflow.


## Build / install

LPM uses the CMake build system. An example configure step is

```
export TPL_DIR=$HOME/lpm-tpl-openmp
cmake -B ./build \
-DCMAKE_CXX_COMPILER=mpic++ \
-DCMAKE_CXX_STANDARD=17 \
-DCMAKE_BUILD_TYPE=RelWithDebInfo \
-DCMAKE_Fortran_COMPILER=mpifort \
-DCMAKE_C_COMPILER=mpicc \
-DLPM_ENABLE_Compose=ON \
-DLPM_ENABLE_DFS=ON \
-DLPM_TPL_DIR=$TPL_DIR \
-DKokkos_DIR=$TPL_DIR \
-DKokkosKernels_DIR=$TPL_DIR \
-DCompadre_DIR=$TPL_DIR \
-DCompose_DIR=$TPL_DIR \
-DSPDLOG_DIR=$TPL_DIR \
-DCATCH2_DIR=$TPL_DIR \
-DVTK_DIR=$TPL_DIR \
-DFFTW3_DIR=$TPL_DIR \
-DFINUFFT_DIR=$TPL_DIR \
-G"Unix Makefiles" . 
```
Note that this file assumes that all TPLs have been installed into the same folder.

Following configuration, LPM may be built with 
```
cmake --build ./build -j 24
```
And tested with `ctest`.

## Examples

Example programs are included in the `lpm/examples` directory; these are the intended user starting points. Other, smaller examples are included in `lpm/tests`, but these are less instructive as their purpose is to ensure that LPM is working properly, not to introduce new users to its API.


