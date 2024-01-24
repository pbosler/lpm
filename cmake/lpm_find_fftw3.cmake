include(CPM)

CPMAddPackage(
  NAME findfftw
  GIT_REPOSITORY "https://github.com/egpbos/findFFTW.git"
  GIT_TAG "master"
  EXCLUDE_FROM_ALL YES
  GIT_SHALLOW YES
)
list(APPEND CMAKE_MODULE_PATH "${findfftw_SOURCE_DIR}")

find_package(FFTW REQUIRED)
message(STATUS "found fftw3: ${FFTW_LIBRARIES}")
set(LPM_USE_FFTW3 TRUE CACHE BOOL "")

if (LPM_USE_FFTW3)
  message(STATUS "FFTW3 libraries enabled.")
endif()

