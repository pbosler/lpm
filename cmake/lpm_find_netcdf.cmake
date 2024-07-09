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
endif(LPM_ENABLE_NETCDF)
