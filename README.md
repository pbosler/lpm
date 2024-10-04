# LPM: Lagrangian Particle methods

![Auto-test](https://github.com/pbosler/lpm/actions/workflows/auto_test.yml/badge.svg)

This file contains several algorithms for computing solutions to PDEs using Lagrangian Particle Methods.  Generally, these methods are "mesh free," in the sense that the numerical methods used to solve the PDE and advance the simulation forward in time do not require a mesh topology to exist between particles.  Optionally, we support meshes that are defined recursively in a quad-tree structure (for 2D problems) for use with adaptive refinement.

## Build / install

LPM uses the CMake build system, and a script `lpm/setup` generates the CMake commands for you. From the `lpm` root directory, run:

- `setup <build_dir>` This creates a script, `config.sh` in the `<build_dir>` directory.
- Edit `config.sh` to your desired options, e.g., third-party library locations, floating point precision, etc.   If your machine does not have LPM's required third party dependencies, LPM will clone those libraries for you and attempt to build them itself.
- Optionally, create a machine file and add it to `lpm/machines`.
- Change to your build directory (`cd <build_dir>`) and execute the configure script (`./config.sh`).  This will run CMake with all of LPM's variables and options pre-configured.
- Build LPM, `make -j`
- Optionally, run the test suite, `make test`
- Optionally, install LPM `make install`

## Examples

Example programs are included in the `lpm/examples` directory; these are the intended user starting points. Other, smaller examples are included in `lpm/tests`, but these are less instructive as their purpose is to ensure that LPM is working properly, not to introduce new users to its API.

## Plotting

tbd.
