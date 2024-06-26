if (FastBVE_DIR)
  message(STATUS "looking for FastBVE at ${FastBVE_DIR}")
  find_package(FastBVE REQUIRED HINTS ${FastBVE_DIR} ${FastBVE_DIR}/lib/cmake)
  set(LPM_USE_FASTBVE ON)
  message(STATUS "... found FastBVE = (1) ${FastBVE} | (2) ${FASTBVE} | (3) ${fastbve} | (4) ${FastBVE_LIBRARIES} | (5) ${FastBVE_FOUND}")
else()
  message(FATAL_ERROR "FastBVE_DIR not specified.")
endif()
