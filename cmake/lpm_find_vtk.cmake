if (LPM_ENABLE_VTK)
 find_package(VTK COMPONENTS
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
 )
 if (NOT VTK_FOUND)
   message("VTK not found.")
 else()
   message(STATUS "Found VTK Version: ${VTK_VERSION}")
   set(LPM_USE_VTK TRUE CACHE BOOL "use vtk")
   message(STATUS "VTK_LIBRARIES = ${VTK_LIBRARIES}")
#   if (VTK_VERSION VERSION_LESS "9.0.0")
#     set(LPM_USE_VTK TRUE)
#   else ()
#     message("lpm is not compatible with this version of VTK.")
#   endif()
 endif()
endif()
