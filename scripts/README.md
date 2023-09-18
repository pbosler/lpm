  # Third-party libraries for LPM
  

  LPM uses Kokkos, and relies on several other libraries that also use Kokkos.  Eventually we'd like to handle this build ourselves but for now we rely on user-supplied TPLs.

  ## Kokkos (Required)
First, we need to install Kokkos so that LPM and all of its other Kokkos-related TPLs can use it.  For now, we still require Kokkos major version 3 (not the recently released 4)because of other TPL dependencies.

These steps and the supplied scripts will walk through a CPU build.  If you're working with GPU hardware, we'll assume you know already enough about Kokkos to get started.

Steps:
1. Clone [Kokkos](https://github.com/kokkos/kokkos) and checkout a version 3.x release (e.g., 3.7.02)
  ```
      git clone git@github.com:kokkos/kokkos.git
      cd kokkos
      git checkout tags/3.7.02
      mkdir build
  ```
2. Copy the lpm script `configure_kokkos_tpl.sh` into your build directory.  Edit the CMAKE_BUILD_TYPE and CMAKE_INSTALL_PREFIX to your liking.  Choose the CPU (Host) parallel model (Threads or OpenMP); on Mac, OpenMP isn't available with the default compilers, so we use threads.

3. Run the configure script, which calls `cmake` for Kokkos, e.g., `./configure_kokkos_tpl.sh`
4. Build it: `make -j 8 && make install`
  
## KokkosKernels and COMPOSE (Required)

Both [KokkosKernels](https://github.com/kokkos/kokkos-kernels) and [Compose](https://github.com/E3SM-Project/COMPOSE) depend only on Kokkos, so they can be built now that Kokkos has finished installation.

### KokkosKernels

Follow the same steps as above: Clone the repo, checkout the latest version 3.x release, copy the configure script and edit it to match your Kokkos configuration & installation details from the previous step.  Run it to finish the CMake configure step.  Then build and install.

### COMPOSE

For Compose, we use a fork instead of the main repo.   Clone that fork, `https://github.com/pbosler/COMPOSE, and checkout the `lpm-tpl` branch.   This branch adds some configuration options that Lpm needs; it changes none of the COMPOSE source.
Again, the same steps apply: copy the configure script, edit it to match your specific machine and installation paths, run it, then build and install.

## Compadre (Required)

[Compadre](https://github.com/sandialabs/compadre) requires both Kokkos and KokkosKernels.   The same steps as above work again here.

## VTK (Optional, but recommended)

VTK is a different kind of build.   It doesn't require Kokkos and it's big enough on its own that LPM will never support its own automatic download-configure-build-install workflow.  However, the `confiugre_vtk_tpl.sh` shows how to configure the VTK-9.2.2 release for use with LPM.

# Ready to build LPM

After the above steps are complete, you'll have a working Kokkos, KokkosKernels, Compadre installation that can be used to build Lpm.  Vtk, too.  An example Lpm configure script, `configure_lpm.sh` is also included.

