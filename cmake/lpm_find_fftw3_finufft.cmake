if (FFTW3_DIR) 
    message(STATUS "LPM: Looking for FFTW3 : ${FFTW3_DIR}")
    include(FindPackageHandleStandardArgs)
#
# 03SEP2025
#
# fftw3 CMake support is experimental only, find_package won't work. 
# we have to do this the hard way.
#
    find_library(FFTW3_DOUBLE_LIB 
        NAMES fftw3 
        PATHS ${FFTW3_DIR}
        PATH_SUFFIXES "lib" "lib64"
        NO_DEFAULT_PATH)
        
    find_library(FFTW3_FLOAT_LIB
        NAMES fftw3f
        PATHS ${FFTW3_DIR}
        PATH_SUFFIXES "lib" "lib64"
        NO_DEFAULT_PATH)
    
    find_library(FFTW3_DOUBLE_OPENMP_LIB
        NAMES fftw3_omp
        PATHS ${FFTW3_DIR}
        PATH_SUFFIXES "lib" "lib64"
        NO_DEFAULT_PATH)
    
    find_library(FFTW3_FLOAT_OPENMP_LIB
        NAMES fftw3f_omp
        PATHS ${FFTW3_DIR}
        PATH_SUFFIXES "lib" "lib64"
        NO_DEFAULT_PATH)
    
    find_path(FFTW3_INCLUDE_DIR 
        NAMES "fftw3.h"
        PATHS ${FFTW3_DIR}
        PATH_SUFFIXES "include"
        NO_DEFAULT_PATH)
    
    if (FFTW3_DOUBLE_LIB)
        add_library(FFTW3::Double INTERFACE IMPORTED)
        set_target_properties(FFTW3::Double PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES ${FFTW3_INCLUDE_DIR}
            INTERFACE_LINK_LIBRARIES ${FFTW3_DOUBLE_LIB})
    else()
        message(FATAL_ERROR "fftw3 library not found")
    endif()
    
    if (FFTW3_FLOAT_LIB) 
        add_library(FFTW3::Float INTERFACE IMPORTED)
        set_target_properties(FFTW3::Float PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES ${FFTW3_INCLUDE_DIR}
            INTERFACE_LINK_LIBRARIES ${FFTW3_FLOAT_LIB})
    else()
        message(FATAL_ERROR "fftw3f library not found")
    endif()
    
    if (FFTW3_DOUBLE_OPENMP_LIB) 
        add_library(FFTW3::DoubleOpenMP INTERFACE IMPORTED)
        set_target_properties(FFTW3::DoubleOpenMP PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES ${FFTW3_INCLUDE_DIR}
            INTERFACE_LINK_LIBRARIES ${FFTW3_DOUBLE_OPENMP_LIB})
    else()
        message(FATAL_ERROR "fftw3_omp library not found")
    endif()
    
    if (FFTW3_FLOAT_OPENMP_LIB) 
        add_library(FFTW3::FloatOpenMP INTERFACE IMPORTED)
        set_target_properties(FFTW3::FloatOpenMP PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES ${FFTW3_INCLUDE_DIR}
            INTERFACE_LINK_LIBRARIES ${FFTW3_FLOAT_OPENMP_LIB})
    else()
        message(FATAL_ERROR "fftw3_omp library not found")
    endif()
    
    find_package_handle_standard_args(FFTW3 
        REQUIRED_VARS FFTW3_INCLUDE_DIR FFTW3_DOUBLE_LIB FFTW3_FLOAT_LIB
        HANDLE_COMPONENTS
    )
else()
    message(FATAL_ERROR "FFTW3_DIR not set")
endif()
    mark_as_advanced(
        FFTW3_INCLUDE_DIR
        FFTW3_DOUBLE_LIB
        FFTW3_FLOAT_LIB
        FFTW3_DOUBLE_OPENMP_LIB
        FFTW3_FLOAT_OPENMP_LIB
    )
set(LPM_USE_FFTW3 TRUE CACHE BOOL "")

#
# 03SEP2025
#
# until finufft improves their own CMake, find_package won't work. 
# we have to do this the hard way.
#
# find_package(finufft REQUIRED HINTS ${FINUFFT_DIR})
#
if (FINUFFT_DIR)
  message(STATUS "LPM: looking for finufft.h at ${FINUFFT_DIR}/include")
  find_path(FINUFFT_INCLUDE_DIR 
    NAMES finufft.h
    DOC "finufft include directory (path to finufft.h)"
    HINTS "${FINUFFT_DIR}/include"
    )
   mark_as_advanced(FINUFFT_INCLUDE_DIR)
   
   message(STATUS "LPM: looking for finufft library at ${FINUFFT_DIR}/${CMAKE_INSTALL_LIBDIR}")
   find_library(FINUFFT_LIBRARY
        NAMES finufft
        DOC "finufft library (libfinufft.a)"
        HINTS "${FINUFFT_DIR}/${CMAKE_INSTALL_LIBDIR}"
    )
    mark_as_advanced(FINUFFT_LIBRARY)
    
    find_package_handle_standard_args(finufft 
        REQUIRED_VARS FINUFFT_INCLUDE_DIR FINUFFT_LIBRARY 
    )
    if (finufft_FOUND)
        if (NOT TARGET finufft::finufft)
            add_library(finufft::finufft UNKNOWN IMPORTED)
            set_target_properties(finufft::finufft PROPERTIES
                IMPORTED_LOCATION ${FINUFFT_LIBRARY}
                INTERFACE_INCLUDE_DIRECTORIES ${FINUFFT_INCLUDE_DIR})
        endif()
    else()
        message(FATAL_ERROR "Cannot find finufft")
    endif()
else()
  message(FATAL_ERROR "FINUFFT_DIR not specified.")
endif()
set(LPM_USE_FINUFFT TRUE CACHE BOOL "finufft found")