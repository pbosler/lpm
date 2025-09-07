# Third-party libraries for LPM
  
  [Spdlog](https://github.com/gabime/spdlog) and [Catch2](https://github.com/catchorg/Catch2) can be built independently, and simply, by following the default instructions for each library.  Some specific configurations that have worked with LPM can be found in the [scripts folder](https://github.com/pbosler/lpm/tree/main/scripts) and in the [tools folder](https://github.com/pbosler/lpm/tree/common-tpls/tools)' Dockerfile.
  
  LPM uses Kokkos, and relies on several other libraries that also use Kokkos.
  
## VTK

VTK may be available from your package manager via, e.g., `sudo apt-get libvtk9-dev`, but if it isn't, some simple [scripts](https://github.com/pbosler/lpm/tree/main/scripts) to configured and build it are included with LPM.
VTK is a large set of software and can take a while to build.

## Kokkos and dependent TPLs

LPM, and several of its dependencies, are written in C++/Kokkos. Kokkos must therefore be installed before KokkosKernels, COMPOSE, and Compadre.  
These steps and the supplied scripts will walk through a CPU build.  If you're working with GPU hardware, we'll assume you know already enough about Kokkos to get started.

### COMPOSE

For Compose, as of 03SEP2025, we need to checkout the `pb-lpm-kokkos-4.7` branch instead of `main`, to allow it to work with the latest release of Kokkos (4.7).

## Optional

To use the double Fourier series (DFS) solver, FFTW3 and FINUFFT are also required.   Note that FINUFFT has non-standard dependencies on FFTW3 that may not be supported by a package manager's `fftw` (specifically, it requires both the `float` and `double` precision `fftw` libraries to be present).   You may have to build FFTW3 yourself to install both libraries side by side; see [FFTW3](fftw3.org) for instructions, and the LPM [tools folder](https://github.com/pbosler/lpm/tree/common-tpls/tools)' Dockerfile for an example.

# Ready to build LPM

After the above steps are complete, you'll have a working Kokkos, KokkosKernels, Compadre installation that can be used to build Lpm.  Vtk, too.  An example Lpm configure script, `configure_lpm.sh` is also included.

