#
# Compose : spherical geometry and shape preservation libraries
#

#set(LPM_USE_COMPOSE FALSE CACHE BOOL "Compose libraries found")
if (LPM_ENABLE_Compose)
  if (Compose_DIR)
    message(STATUS "looking for Compose at ${Compose_DIR}")
    set(Compose_INCLUDE_DIR ${Compose_DIR}/include)
    set(Compose_LIBRARY_DIR ${Compose_DIR}/lib)
    set(Compose_LIBRARY ${Compose_LIBRARY_DIR}/libcompose.a)
    if (NOT EXISTS ${Compose_INCLUDE_DIR})
      message(FATAL_ERROR "Could not find Compose include dir at ${Compose_INCLUDE_DIR}")
    else()
      if (NOT EXISTS ${Compose_LIBRARY})
        message(FATAL_ERROR "Could not find Compose library at ${Compose_LIBRARY}")
      else()
        add_library(compose STATIC IMPORTED GLOBAL)
        set_target_properties(compose PROPERTIES IMPORTED_LOCATION ${Compose_LIBRARY})
        list(APPEND LPM_EXT_INCLUDE_DIRS ${Compose_INCLUDE_DIR})
#        list(APPEND LPM_EXT_LIBRARIES compose)
        set(LPM_USE_COMPOSE TRUE CACHE BOOL "Compose libraries found")
      endif()
    endif()
  else()
    message(FATAL_ERROR "Compose_DIR required.")
  endif()
  if (LPM_USE_COMPOSE)
    message(STATUS "Compose libraries enabled.")
  endif()
endif(LPM_ENABLE_Compose)
