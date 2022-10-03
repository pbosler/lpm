#
# spdlog: Required
#
#list(APPEND CMAKE_MODULE_PATH $ENV{SPDLOG_ROOT})
#find_package(spdlog CONFIG HINTS $ENV{SPDLOG_ROOT})
#
#if (spdlog_FOUND)
#  message(STATUS "spdlog found: ${spdlog_INCLUDE_DIR}")
#  message(STATUS "spdlog libraries:  ${spdlog_LIBRARIES}")
#  message(STATUS "spdlog_DIR:  ${spdlog_DIR}")
#  message(STATUS "spdlog_INTERFACE_INCLUDE_DIRECTORIES: ${spdlog_INTERFACE_INCLUDE_DIRECTORIES}")
#  list(APPEND LPM_EXT_INCLUDE_DIRS ${spdlog_DIR}/../../../include)
#else()
#  message(FATAL_ERROR "cannot find spdlog")
#endif()
include(ExternalProject)
include(GNUInstallDirs)

set(SPDLOG_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")
set(SPDLOG_LIBRARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(SPDLOG_LIBRARY "${SPDLOG_LIBRARY_DIR}/libspdlog.a")
message(STATUS "Building spdlog library: ${SPDLOG_LIBRARY}")
set(LPM_NEEDS_SPDLOG_BUILD TRUE)

add_library(spdlog STATIC IMPORTED GLOBAL)
set_target_properties(spdlog PROPERTIES IMPORTED_LOCATION ${SPDLOG_LIBRARY})
list(APPEND LPM_EXT_INCLUDE_DIRS ${SPDLOG_INCLUDE_DIR})
set(LPM_EXT_LIBRARIES spdlog;${LPM_EXT_LIBRARIES})
set(SPDLOG_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
if (LPM_NEEDS_SPDLOG_BUILD)
 set(SPDLOG_CMAKE_OPTS -DCMAKE_INSTALL_PREFIX=${SPDLOG_INSTALL_PREFIX}
                     -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
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
