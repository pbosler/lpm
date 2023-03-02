include(FortranCInterface)
FortranCInterface_VERIFY(CXX)

FortranCInterface_HEADER(lpm_fortran_c.h
                         SYMBOLS trmesh
                                 getsig
                                 intrc1
                                 gradl)

add_library(lpm_fortran STATIC
                        bivar.f90
                        stripack.f
                        ssrfpack.f)
target_include_directories(lpm_fortran  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR})

list(APPEND LPM_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
                             ${CMAKE_CURRENT_BINARY_DIR} PARENT_SCOPE)

