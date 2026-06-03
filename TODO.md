# LPM open work items

Living list of in-progress / outstanding work that isn't documented elsewhere.
Items are intentionally ordered by priority.

## 1. Numerical blow-up in the polar vortex example (highest priority)

The earlier segfault is fixed (it was a NaN from the unguarded `tan(theta)`
singularity in `JM86Forcing::forcing_b`/`forcing_bprime` at the equator). With
the NaN gone, the polar vortex example now exposes a numerical instability: the
relative vorticity explodes from O(1) to O(1e6) in a single timestep, the
particles disperse, and the next step's mesh->grid GMLS stalls.

Plan: change the time integrator so that at each RK stage the relative vorticity
is computed directly from the (invariant) absolute vorticity — i.e. carry/advect
absolute vorticity and recover relative vorticity as
`rel_vort = abs_vort - coriolis` each stage — rather than time-integrating the
relative vorticity tendency as the current `DFSPolarVortexRK4` does
(`src/dfs/lpm_dfs_polar_vortex_solver_impl.hpp`).

## 2. Refactor `src/dfs/` view allocations

The DFS code allocates `Kokkos::View`s inside functions that are called once
per time step. These should be hoisted into longer-lived storage on the owning
solver class so the allocator isn't hit every step.

## 3. Resolve duplicate `liblpm.a` link warnings

The linker emits `ld: warning: ignoring duplicate libraries: '../src/liblpm.a'`
for several targets. Harmless (the duplicate is ignored), but it points at
`lpm` being listed more than once on a link line — likely `lpm` plus
`${LPM_LIBRARIES}` both resolving to `liblpm.a`. Deduplicate the link
dependencies in the relevant `target_link_libraries` calls to silence it.

## 4. Fix `-Wwritable-strings` warnings in `lpm_input_test.cpp`

`tests/lpm_input_test.cpp:107` builds `char* argv[]` directly from string
literals, which is ill-formed in C++11 (`-Wwritable-strings`). Change the array
to `const char*` (and adjust the `parse_args` call/signature as needed), or use
modifiable `char` buffers, so the test stops relying on the non-conforming
conversion.

## 5. Finish the `Lpm*` → `lpm_*` migration

Each remaining `LpmFoo.{hpp,cpp}` file needs to be either refactored into the
new snake_case style (with `lpm_foo.hpp` / `lpm_foo.cpp` / `lpm_foo_impl.hpp`
split) or deleted if obsolete. See CLAUDE.md's "Two file-naming generations"
note for context. `src/CMakeLists.txt` is the source of truth for which legacy
files are still built into `liblpm`.
