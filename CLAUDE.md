# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

LPM (Lagrangian Particle Methods) is a C++17 + Kokkos library for solving PDEs (barotropic vorticity, shallow water, 2D transport) with mesh-free particle methods on the plane and the sphere. Optional quad-tree meshes support adaptive refinement. MPI is required; OpenMP and CUDA are reached through Kokkos.

## Build

Configuration is via CMake. All required third-party libraries (TPLs) must be pre-built and discoverable. The repo includes `configure-lpm-openmp.sh` at the root as a working example; it expects TPLs under `$HOME/lpm-tpl-openmp`. Typical flow:

```
cmake -B ./build \
  -DCMAKE_CXX_COMPILER=mpic++ -DCMAKE_C_COMPILER=mpicc -DCMAKE_Fortran_COMPILER=mpifort \
  -DCMAKE_CXX_STANDARD=17 -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLPM_ENABLE_Compose=ON -DLPM_ENABLE_DFS=ON \
  -DKokkos_DIR=$TPL_DIR -DKokkosKernels_DIR=$TPL_DIR \
  -DCompadre_DIR=$TPL_DIR -DCompose_DIR=$TPL_DIR \
  -DSPDLOG_DIR=$TPL_DIR -DCATCH2_DIR=$TPL_DIR -DVTK_DIR=$TPL_DIR \
  -DFFTW3_DIR=$TPL_DIR -DFINUFFT_DIR=$TPL_DIR \
  -G"Unix Makefiles" .
cmake --build ./build -j 24
```

Build options that gate large code regions:
- `LPM_ENABLE_DFS` — double Fourier series solver on the sphere; pulls in `src/dfs/`, FFTW3 and finufft. CI runs both ON and OFF.
- `LPM_ENABLE_Compose` — COMPOSE semi-Lagrangian property preservation.
- `LPM_ENABLE_NETCDF` — gates `src/netcdf/` sources and the `netcdf_test`.
- `LPM_USE_VTK` — gates `src/vtk/` sources; default ON.

`LpmConfig.h` is generated from `LpmConfig.h.in` at configure time. It defines the central typedefs (`Index = int`, `Real = double`, `Int`, `Uint`, `Short`, optional `Complex`) inside `namespace Lpm`, plus `LPM_MESH_SEED_DIR` and `LPM_TEST_DATA_DIR` paths that the binaries read at runtime — so tests and examples must be run against the build tree that configured them.

## Tests

Run all tests with `ctest` from the build dir. Test executables live in `build/tests/`.

Tests are registered via the `CreateUnitTest()` macro in `cmake/create_unit_test.cmake`. It generates one CTest entry per (MPI rank count, OMP thread count) pair, named `{target}_ut_np{N}_omp{T}`. Examples:
- `ctest -R polymesh_test` — run all polymesh test variants
- `ctest -R faces_test_ut_np1_omp1` — exact one configuration
- `./tests/polymesh_test [tag]` — Catch2 binary; pass tag/name filters directly
- `ctest -V` — verbose; mirrors what CI runs

Test ranges are governed by cache vars `LPM_TEST_MAX_RANKS` (default 2) and `LPM_TEST_MAX_THREADS` (default `$OMP_NUM_THREADS`). MPI tests inside the CI Docker container need `DOCKER_ALLOW_MPI_RUN_AS_ROOT=1` and `DOCKER_ALLOW_MPI_RUN_AS_ROOT_CONFIRM=1` — these are set in the workflow.

Most tests use the shared `lpm_test_main` micro-library (a precompiled Catch2 main, see `src/util/lpm_catch_main.cpp`). Pass `EXCLUDE_CATCH_MAIN` to `CreateUnitTest` to provide your own main.

DFS tests (`dfs_grid_test`, `dfs_bve_test`) and `netcdf_test` only register when the corresponding `LPM_ENABLE_*` flag is ON.

## Code style

- `.clang-format` is Google base with left pointer alignment and aligned consecutive assignments/macros. Run before committing — recent merges have been clang-format-only cleanups.
- C++17, `CMAKE_CXX_EXTENSIONS OFF` (required by Kokkos).
- Everything lives in `namespace Lpm`.

## Architecture

### Two file-naming generations

The codebase is mid-transition between conventions. Both are present and both compile in.
- **Newer (`lpm_*` snake_case)** — the active layer. New code goes here. Header is `lpm_foo.hpp`, source is `lpm_foo.cpp`, templated definitions go in `lpm_foo_impl.hpp` and are included by headers that need them.
- **Older (`Lpm*` PascalCase)** — `LpmOctree*`, `LpmNodeArray*`, `LpmBVEKernels`, `LpmSWE*`, `LpmShallowWater*`, etc. Mostly headers; many are not in the current `LPM_SOURCES` list and aren't built into `liblpm`. Treat them as legacy/reference unless you confirm otherwise via `src/CMakeLists.txt`.

When adding files, follow the new convention and register sources in `src/CMakeLists.txt`.

### Source layout (`src/`)

- root — top-level model classes (`lpm_bve_sphere`, `lpm_incompressible2d`, `lpm_2d_transport_*`), RK time integrators (`*_rk4`, `*_rk2`), shared infra (`lpm_coords`, `lpm_field`, `lpm_geometry`, `lpm_comm`, `lpm_logger`, `lpm_input`, `lpm_compadre`), and `lpm_constants` / `lpm_assert`.
- `mesh/` — `PolyMesh2d` (the main mesh type) plus `Vertices`, `Edges`, `Faces`, `MeshSeed`, remeshing (`lpm_compadre_remesh`, `lpm_bivar_remesh`), gather/scatter, refinement.
- `tree/` — CPU quadtree and `Box3d` for AMR.
- `util/` — Catch2 main, timer, logger plumbing, math, I/O helpers (matlab, numpy), string/tuple/stl.
- `fortran/` — Fortran interpolation libraries (`bivar`, `ssrfpack`, `stripack`) with C++ wrappers (`lpm_bivar_interface`, `lpm_ssrfpack_interface`). Built as separate `lpm_fortran` static lib.
- `dfs/` — Double Fourier series solver for the sphere (BVE + SWE). Only compiled when `LPM_ENABLE_DFS=ON`.
- `netcdf/`, `vtk/` — I/O backends, each gated by their own flag.

### Mesh model

`PolyMesh2d<SeedType>` is templated on a `MeshSeed` type that bundles geometry (`PlaneGeometry` or `SphereGeometry` from `lpm_geometry.hpp`), face kind (`TriFace`/`QuadFace`/`VoronoiFace`), and an on-disk seed file in `mesh_seeds/` (e.g. `quadRectSeed.dat`, `icosTriSphereSeed.dat`, `cubedSphereSeed.dat`). `LPM_MESH_SEED_DIR` (set in `LpmConfig.h`) points at the source-tree mesh_seeds folder at runtime. `meshSeeds.py` regenerates the `.dat` files.

The seed defines tree level 0; refinement adds depth via the quad-tree, optionally with AMR limits. `PolyMeshParameters` controls initial depth, radius, and AMR memory headroom.

### Kokkos conventions (see `doc/Design.md`)

1. All arrays are `Kokkos::View`.
2. Mesh topology mutations (refinement, coarsening, reconnection) happen on the host only; device views are updated via `deep_copy`.
3. Device-resident "main" views are public on the containing class; host mirror views are private.
4. `MeshSeed` owns geometry and face-kind information — pick the right seed type and you get the right geometry for free.

### Examples vs tests

- `examples/` — the intended starting point for new users. Each is a self-contained driver for one problem (BVE Rossby-Haurwitz, gravity waves, transport, shallow water TC2, …). Read these to see how the pieces compose.
- `tests/` — Catch2 unit tests, organized per component. Many are useful as miniature usage examples but are written for coverage, not pedagogy.

## CI

`.github/workflows/auto_test.yml` builds inside `pbosler/lpm-tpl-sep25:0` (TPLs pre-installed under `/tpl`) across `Debug`/`RelWithDebInfo` x `LPM_ENABLE_DFS=ON`/`OFF`. NETCDF is OFF in CI. Test runner uses `OMP_NUM_THREADS=4`, `OMP_PROC_BIND=spread`, `OMP_PLACES=threads`. `ctest -V --no-tests=error` is the pass/fail gate.

The TPL Docker image is built from `tools/Dockerfile`; see `tools/README.md` for the current pinned versions (Kokkos 4.7.00, Compadre 1.6.2, COMPOSE branch `pb-lpm-kokkos-4.7`, FFTW 3.3.10, finufft 2.4.1 as of Sep 2025).
