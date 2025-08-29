message(STATUS "Looking for VTK at $ENV{VTK_ROOT} ${VTK_DIR}")
find_package(VTK REQUIRED COMPONENTS
    CommonColor
    CommonCore
    CommonDataModel
    IOCore
    IOGeometry
    IOImage
    IOLegacy
    IOParallel
    IOXML
    IOXMLParser
    ParallelCore
    ParallelMPI
    HINTS $ENV{VTK_ROOT} ${VTK_DIR}
)

message(STATUS "Found VTK Version: ${VTK_VERSION}")

message(STATUS "VTK_LIBRARIES = ${VTK_LIBRARIES}")
if (VTK_VERSION VERSION_LESS "9.0.0")
    set(LPM_USE_VTK TRUE)
else ()
    message("lpm is not compatible with this version of VTK.")
endif()

