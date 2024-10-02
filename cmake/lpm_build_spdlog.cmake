include(ExternalProject)
include(GNUInstallDirs)

define_property(GLOBAL PROPERTY SPDLOG_BUILT
  BRIEF_DOCS "Whether spdlog has been built"
  FULL_DOCS "Used to ensure spdlog only gets built once.")

get_property(IS_SPDLOG_BUILT GLOBAL PROPERTY SPDLOG_BUILT SET)

if (NOT IS_SPDLOG_BUILT)

  set(SPDLOG_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})

  set(SPDLOG_INCLUDE_DIR "${SPDLOG_INSTALL_PREFIX}/include")
  set(SPDLOG_LIBRARY_DIR "${SPDLOG_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
  set(SPDLOG_LIBRARY "${SPDLOG_LIBRARY_DIR}/libspdlog.a")
  message(STATUS "Building spdlog library: ${SPDLOG_LIBRARY}")

  add_library(spdlog STATIC IMPORTED GLOBAL)
  set_target_properties(spdlog PROPERTIES IMPORTED_LOCATION ${SPDLOG_LIBRARY})
  list(APPEND LPM_EXT_INCLUDE_DIRS ${SPDLOG_INCLUDE_DIR})

  set(SPDLOG_CMAKE_OPTS -DCMAKE_INSTALL_PREFIX=${SPDLOG_INSTALL_PREFIX}
                     -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                     -DCMAKE_BUILD_TYPE=RelWithDebInfo
                     -DBUILD_SHARED_LIBS=OFF
                     -DSPDLOG_BUILD_EXAMPLE=OFF
                     -DSPDLOG_BUILD_TESTS=OFF
                     -DSPDLOG_ENABLE_PCH=ON
                     -DSPDLOG_INSTALL=ON
                     )

  ExternalProject_Add(spdlog_proj
         PREFIX ${CMAKE_CURRENT_BINARY_DIR}/ext/spdlog
         SOURCE_DIR ${PROJECT_SOURCE_DIR}/ext/spdlog
         BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/ext/spdlog
         INSTALL_DIR ${SPDLOG_INSTALL_PREFIX}
         CMAKE_ARGS ${SPDLOG_CMAKE_OPTS}
         BUILD_COMMAND ${MAKE} -j 8
         LOG_CONFIGURE TRUE
         LOG_BUILD TRUE
         LOG_INSTALL TRUE
         )
  add_dependencies(spdlog spdlog_proj)




endif()
