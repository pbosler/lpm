#INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/src"
#                    "${PROJECT_BINARY_DIR}"
#                    "${VTK_INCLUDE_DIRS}"
##                   "${Trilinos_INCLUDE_DIRS}"
##                   "${Trilinos_TPL_INCLUDE_DIRS}"
#)
#
#ADD_EXECUTABLE(sphereInit SphereMeshInit.cpp)
#TARGET_LINK_LIBRARIES(sphereInit lpm ${Trilinos_LIBRARIES} ${VTK_LIBRARIES} ${Trilinos_TPL_LIBRARIES})
#
#ADD_EXECUTABLE(spherePoisson SphereVorticity.cpp)
#TARGET_LINK_LIBRARIES(spherePoisson lpm ${Trilinos_LIBRARIES} ${VTK_LIBRARIES} ${Trilinos_TPL_LIBRARIES})
#
