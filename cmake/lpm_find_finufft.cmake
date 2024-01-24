include(CPM)

printvar(LPM_USE_OPENMP)
set(FINUFFT_OPTIONS "FINUFFT_USE_OPENMP ${LPM_USE_OPENMP}")

CPMAddPackage(
  NAME             Finufft
  GIT_REPOSITORY   https://github.com/flatironinstitute/finufft.git
  GIT_TAG          master
  GIT_SHALLOW      Yes
  GIT_PROGRESS     Yes
  EXCLUDE_FROM_ALL Yes
  OPTIONS          ${FINUFFT_OPTIONS}
  SYSTEM
)

set(LPM_USE_FINUFFT TRUE CACHE BOOL "")
