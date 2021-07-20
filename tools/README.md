# LPM Tools

This folder contains tools used by LPM for constructing metadata about the status of the repository and building third-party libraries (TPL).

## Docker images

To facilitate auto-testing via github actions, we use a docker image to contain pre-built TPLs.
To use the docker file, you must first download the tarballs for the TPLs you want to build.
Currently, all of them.  If you don't want to build a TPL in a docker image, simply comment out that portion of the docker file.

Run the docker image build process by executing the `build-ext-docker-image.sh` script from your local shell.  You must have Docker installed locally.

TPLs in the docker image:
- CMake 3.21: Trilinos (required by LPM) requires CMake version >= 3.18, which are not yet available from the ubuntu package manager, so we build it ourselves.
- HDF5-1.12.1: A working HDF5 installation is required for netCDF.  This release works.
- netCDF-4.8.0: Required by LPM for restart and data sharing/archiving capabilities.  This is the latest release in July 2021, when this docker file was made.
- VTK-8.1.2: Required by LPM for some data storage capabilities; may be made optional later.  Versions > 8 have some changes that are not backwards compatible so currently LPM requires VTK versions < 9.  This one works on all of LPM's test platforms.  Version 8.2 has also been tested successfully.
- Trilinos-13.0.1: Required by LPM for meshfree functional approximations via generalized moving least squares methods (GMLS), and for parallel domain decomposition via MPI. This release is the latest stable release in July 2021, when the docker file was made.
- Yaml-cpp-0.6.3: Required by Trilinos and soon-to-be required by LPM.  This release is the latest stable release available from the SNLComputation github organization as of July 2021.
- spdlog-1.8.5: Required by LPM for console & file logging capabilities.  1.8.5 is the latest stable release as of July 2021.

**Note:** Any TPLs not contained in the docker image will be cloned from their respective github repositories by LPM automatically (via `git submodule` commands) and LPM will attempt to build them as part of its build process.


