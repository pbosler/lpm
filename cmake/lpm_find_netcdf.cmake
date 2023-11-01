#
# NetCDF (Optional)
#
set(LPM_USE_NETCDF FALSE)
if (LPM_ENABLE_NETCDF)

  # look for a CMake build of netcdf
  find_package(netCDF CONFIG QUIET)
  if (netCDF_FOUND)
    set(NetCDF_FOUND "${netCDF_FOUND}")
    set(NetCDF_INCLUDE_DIRS "${netCDF_INCLUDE_DIR}")
    set(NetCDF_LIBRARIES "${netCDF_LIBRARIES}")
    set(NetCDF_VERSION "${NetCDFVersion}")

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(NetCDF
      REQUIRED_VARS NetCDF_INCLUDE_DIRS NetCDF_LIBRARIES
      VERSION_VAR NetCDF_VERSION)

    if (NOT TARGET NetCDF::NetCDF)
      add_library(NetCDF::NetCDF INTERFACE IMPORTED)
      if (TARGET "netCDF::netcdf")
        set_target_properties(NetCDF::NetCDF PROPERTIES
          INTERFACE_LINK_LIBRARIES "netCDF::netcdf")
      elseif (TARGET "netcdf")
        set_target_properties(NetCDF::NetCDF PROPERTIES
          INTERFACE_LINK_LIBRARIES "netcdf")
      else()
        set_target_properties(NetCDF::NetCDF PROPERTIES
          INTERFACE_LINK_LIBRARIES "${NetCDF_LIBRARIES}")
      endif()
    endif()

  else()
    # look for a pgkconfig build of netcdf
    find_package(PkgConfig QUIET)
    if (PkgConfig_FOUND)
      pkg_check_modules(_NetCDF QUIET netcdf IMPORTED_TARGET)

      if (_NetCDF_FOUND)
        message(STATUS "Found netcdf pkg")
        set(NetCDF_FOUND "${_NetCDF_FOUND}")
        set(NetCDF_INCLUDE_DIRS "${_NetCDF_INCLUDE_DIRS}")
        set(NetCDF_LIBRARIES "${_NetCDF_LIBRARIES}")
        set(NetCDF_VERSION "${_NetCDF_VERSION}")

        include(FindPackageHandleStandardArgs)
        find_package_handle_standard_args(NetCDF
          REQUIRED_VARS NetCDF_LIBRARIES
          VERSION_VAR NetCDF_VERSION)

        if (NOT TARGET NetCDF::NetCDF)
          add_library(NetCDF::NetCDF INTERFACE IMPORTED)
          set_target_properties(NetCDF::NetCDF PROPERTIES
            INTERFACE_LINK_LIBRARIES "PkgConfig::_NetCDF")
        endif()

      else()
        message(STATUS "looking for netcdf ...")
        # look for a manually installed build of netcdf
        find_path(NetCDF_INCLUDE_DIR
          NAMES netcdf.h
          DOC "netcdf include directory"
          HINTS  "$ENV{NETCDF_ROOT}/include"
        )
        mark_as_advanced(NetCDF_INCLUDE_DIR)

        find_library(NetCDF_LIBRARY
          NAMES netcdf
          DOC "netcdf library"
          HINTS "$ENV{NETCDF_ROOT}/lib" "$ENV{NETCDF_ROOT}/lib64"
        )
        mark_as_advanced(NetCDF_LIBRARY)

        if (NetCDF_INCLUDE_DIR)
          file(STRINGS "${NetCDF_INCLUDE_DIR}/netcdf_meta.h" _netcdf_version_lines
            REGEX "#define[ \t]+NC_VERSION_(MAJOR|MINOR|PATCH|NOTE)")
          string(REGEX REPLACE ".*NC_VERSION_MAJOR *\([0-9]*\).*" "\\1" _netcdf_version_major "${_netcdf_version_lines}")
          string(REGEX REPLACE ".*NC_VERSION_MINOR *\([0-9]*\).*" "\\1" _netcdf_version_minor "${_netcdf_version_lines}")
          string(REGEX REPLACE ".*NC_VERSION_PATCH *\([0-9]*\).*" "\\1" _netcdf_version_patch "${_netcdf_version_lines}")
          string(REGEX REPLACE ".*NC_VERSION_NOTE *\"\([^\"]*\)\".*" "\\1" _netcdf_version_note "${_netcdf_version_lines}")
          set(NetCDF_VERSION "${_netcdf_version_major}.${_netcdf_version_minor}.${_netcdf_version_patch}.${_netcdf_version_note}")

          unset(_netcdf_version_major)
          unset(_netcdf_version_minor)
          unset(_netcdf_version_patch)
          unset(_netcdf_version_note)
          unset(_netcdf_version_lines)
        endif()

        include(FindPackageHandleStandardArgs)
        find_package_handle_standard_args(NetCDF
          REQUIRED_VARS NetCDF_LIBRARY NetCDF_INCLUDE_DIR
          VERSION_VAR NetCDF_VERSION)

        if (NetCDF_FOUND)
          set(NetCDF_INCLUDE_DIRS "${NetCDF_INCLUDE_DIR}")
          set(NetCDF_LIBRARIES "${NetCDF_LIBRARY}")

          if (NOT TARGET NetCDF::NetCDF)
            add_library(NetCDF::NetCDF UNKNOWN IMPORTED)
            set_target_properties(NetCDF::NetCDF PROPERTIES
              IMPORTED_LOCATION "${NetCDF_LIBRARY}"
              INTERFACE_INCLUDE_DIRECTORIES "${NetCDF_INCLUDE_DIR}")
          endif()
        else()
          message(FATAL_ERROR "Cannot find NetCDF.")
        endif()
      endif()
    endif()
  endif()
  set(LPM_USE_NETCDF TRUE CACHE BOOL "NetCDF found.")

#  find_package(ZLIB REQUIRED)
#   #
#   #  look for CMake-built HDF5
#   #
#  list(APPEND CMAKE_MODULE_PATH $ENV{HDF5_ROOT})
#  find_package(hdf5 CONFIG QUIET)
#  if (hdf5_FOUND)
#     message(STATUS "found hdf5")
#  else(hdf5_FOUND)
#    #
#    #   look for non-CMake HDF5
#    #
#    message(STATUS "cmake package hdf5 not found. Looking for explicit include dirs and libraries...")
#    if(HDF5_INCLUDE_DIR)
#      if (NOT EXISTS ${HDF5_INCLUDE_DIR})
#        message(FATAL_ERROR "Couldn't find HDF5 include dir at ${HDF5_INCLUDE_DIR}.")
#      endif()
#      message(STATUS "looking for hdf5 headers in include dir: ${HDF5_INCLUDE_DIR}.")
#      if (EXISTS ${HDF5_INCLUDE_DIR}/hdf5.h)
#        message(STATUS "found hdf5.h")
#        if (EXISTS ${HDF5_INCLUDE_DIR}/hdf5_hl.h)
#          message(STATUS "found hdf5_hl.h")
#        else()
#          message(FATAL_ERROR "cannof find hdf5_hl.h")
#        endif()
#      else()
#        message(FATAL_ERROR "cannot find hdf5.h")
#      endif()
#      if (HDF5_LIBRARY)
#        if (NOT EXISTS ${HDF5_LIBRARY})
#          message(FATAL_ERROR "Couldn't find hdf5 library at ${HDF5_LIBRARY}.")
#        endif()
#        message(STATUS "found hdf5 library at ${HDF5_LIBRARY}.")
#      else(HDF5_LIBRARY)
#        message(FATAL_ERROR "on a system without a CMake package for hdf5, the variables HDF5_LIBRARY, must be set to the full path of libhdf5.a or libhdf5.so")
#      endif(HDF5_LIBRARY)
#      if (HDF5_HL_LIBRARY)
#        if (NOT EXISTS ${HDF5_HL_LIBRARY})
#          message(FATAL_ERROR "Couldn't find high-level hdf5 library at ${HDF5_HL_LIBRARY}.")
#        endif()
#        message(STATUS "found high-level hdf5 library at ${HDF5_HL_LIBRARY}.")
#      else (HDF5_HL_LIBRARY)
#        message(FATAL_ERROR "on a system without a CMake package for hdf5, the variables HDF5_HL_LIBRARY, must be set to the full path of libhdf5_hl.a or libhdf5_hl.so")
#      endif(HDF5_HL_LIBRARY)
#    else(HDF5_INCLUDE_DIR)
#      message(FATAL_ERROR "on a system without a CMake package for hdf5, the variables HDF5_INCLUDE_DIR, must be set")
#    endif(HDF5_INCLUDE_DIR)
#
#    set(hdf5_FOUND TRUE)
#    if (HDF5_LIBRARY MATCHES ".so")
#      add_library(hdf5 SHARED IMPORTED GLOBAL)
#      add_library(hdf5_hl SHARED IMPORTED GLOBAL)
#    else()
#      add_library(hdf5 STATIC IMPORTED GLOBAL)
#      add_library(hdf5_hl STATIC IMPORTED GLOBAL)
#    endif()
#    set_target_properties(hdf5 PROPERTIES IMPORTED_LOCATION ${HDF5_LIBRARY})
#    set_target_properties(hdf5_hl PROPERTIES IMPORTED_LOCATION ${HDF5_HL_LIBRARY})
#    list(APPEND LPM_EXT_INCLUDE_DIRS ${HDF5_INCLUDE_DIR})
#    list(APPEND LPM_EXT_LIBRARIES hdf5 hdf5_hl)
#  endif(hdf5_FOUND)
#
#  list(APPEND CMAKE_MODULE_PATH $ENV{NETCDF_ROOT})
#  find_package(netcdf CONFIG QUIET)
#  if (netcdf_FOUND)
#    message(STATUS "found netcdf cmake package.")
#    printvar(netcdf_INCLUDE_DIRS)
#    printvar(netcdf_LIBRARIES)
#  else (netcdf_FOUND)
#    #
#    # look for non-cmake netcdf
#    #
#    message(STATUS "cmake package netcdf not found.  Looking for explicit include dirs and library")
#    if (NETCDF_INCLUDE_DIR)
#      if (NOT EXISTS ${NETCDF_INCLUDE_DIR})
#        message(FATAL_ERROR "Couldn't find NetCDF include dir at ${NETCDF_INCLUDE_DIR}.")
#      endif()
#      message(STATUS "looking for netcdf.h in include dir: ${NETCDF_INCLUDE_DIR}.")
#      if (EXISTS ${NETCDF_INCLUDE_DIR}/netcdf.h)
#        message(STATUS "found netcdf.h")
#      else()
#        message(FATAL_ERROR "cannot find netcdf.h")
#      endif()
#    else(NETCDF_INCLUDE_DIR)
#      message(FATAL_ERROR "on systems with non-cmake netcdf, the variable NETCDF_INCLUDE_DIR must be set to the path of netcdf.h")
#    endif(NETCDF_INCLUDE_DIR)
#    if (NETCDF_LIBRARY)
#      if (NOT EXISTS ${NETCDF_LIBRARY})
#        message(FATAL_ERROR "Couldn't find NetCDF library at ${NETCDF_LIBRARY}.")
#      endif()
#      message(STATUS "found netCDF library at ${NETCDF_LIBRARY}.")
#    else(NETCDF_LIBRARY)
#      message(STATUS "on systems with non-cmake netdcdf, the variable NETCDF_LIBRARY must be set to the full path of libnetcdf.a or libnetcdf.so")
#    endif(NETCDF_LIBRARY)
#
#    #
#    #  if we've made it this far, we've found the all the components we need
#    #  to use a netcdf that was built without cmake.
#    #
#    set(netcdf_FOUND TRUE)
#    set(LPM_USE_NETCDF TRUE CACHE BOOL "netcdf: all components are found.")
#    if (LPM_USE_NETCDF)
#      message(STATUS "netcdf is enabled, and all components are found.")
#    endif()
#    if (NETCDF_LIBRARY MATCHES ".so")
#      add_library(netcdf SHARED IMPORTED GLOBAL)
#    else()
#      add_library(netcdf STATIC IMPORTED GLOBAL)
#    endif()
#    set_target_properties(netcdf PROPERTIES IMPORTED_LOCATION ${NETCDF_LIBRARY})
#    list(APPEND LPM_EXT_INCLUDE_DIRS ${NETCDF_INCLUDE_DIR})
#    list(APPEND LPM_EXT_LIBRARIES netcdf)
#  endif(netcdf_FOUND)
endif(LPM_ENABLE_NETCDF)
