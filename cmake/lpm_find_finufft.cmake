include(CPM)

printvar(LPM_USE_OPENMP)
set(FINUFFT_OPTIONS "FINUFFT_USE_OPENMP ${LPM_USE_OPENMP}"
                    "FINUFFT_ENABLE_SANITIZERS OFF")

CPMAddPackage(
  NAME             finufft
  GIT_REPOSITORY   https://github.com/flatironinstitute/finufft.git
  GIT_TAG          v2.3.0
  GIT_SHALLOW      Yes
  GIT_PROGRESS     Yes
  EXCLUDE_FROM_ALL Yes
  OPTIONS          ${FINUFFT_OPTIONS}
)

set(LPM_USE_FINUFFT TRUE CACHE BOOL "")
printvar(LPM_USE_FINUFFT)
