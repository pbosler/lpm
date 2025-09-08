# Misc helper functions for CMake

function (printvar var)
    message("${var}: ${${var}}")
endfunction()

function (print_imported)
  get_property(imported_tgts DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY IMPORTED_TARGETS)
  printvar(imported_tgts)
endfunction()

