message(STATUS "LPM: Looking for VTK at $ENV{VTK_ROOT} ${VTK_DIR}")
find_package(VTK 9 REQUIRED COMPONENTS
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

message(STATUS "LPM: Found VTK Version: ${VTK_VERSION}")

