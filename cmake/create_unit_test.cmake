#
# Most of these CMake functions were written for the E3SM Kokkos Application Toolket (EKAT)
#     https://github.com/E3SM-Project/EKAT
# Please see that repository for these functions' authors.
#

include(CMakeParseArguments)

macro(CheckArgs macro_name parse_prefix valid_options valid_one_value_options valid_multivalue_options)
  if (${parse_prefix}_UNPARSED_ARGUMENTS)
    message(AUTHOR_WARNING
      "Warning: the following arguments to ${macro_name} were not recognized:\n"
      "    ${${parse_prefix}_UNPARSED_ARGUMENTS}\n"
      "  Here is a list of valid arguments:\n"
      "    options: ${valid_one_value_options}\n"
      "    one_value_args: ${valid_one_value_options}\n"
      "    multivalue_args: ${valid_multivalue_options}\n")
  endif()

  if (${parse_prefix}_KEYWORDS_MISSING_VALUES)
    message(AUTHOR_WARNING
      "Warning: the following keywords in macro ${macro_name} were used, but no argument was provided:\n"
      "   ${${parse_prefix}_KEYWORDS_MISSING_VALUES}\n")
  endif()
endmacro()

function(CreateUnitTest target_name target_sources)

  #---------------------------#
  #   Parse function inputs   #
  #---------------------------#
  set(options EXCLUDE_CATCH_MAIN SERIAL THREADS_SERIAL RANKS_SERIAL)
  set(oneValueArgs DEP MPI_EXEC_NAME MPI_NP_FLAG)
  set(multiValueArgs MPI_RANKS THREADS
          MPI_EXTRA_ARGS EXE_ARGS
          INCLUDE_DIRS
          COMPILER_DEFS
          COMPILER_FLAGS
          LIBS LIBS_DIRS LINKER_FLAGS
          LABELS
          PROPERTIES )
  cmake_parse_arguments(lpmtest "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  CheckArgs(CreateTest lpmtest "${options}" "${oneValueArgs}" "${multiValueArgs}")

  # Strip leading/trailing whitespaces from some vars, to avoid either cmake errors
  # (e.g., in target_link_libraries) or compiler errors (e.g. if COMPILER_DEFS=" ")
  string(STRIP "${lpmtest_LIBS}" lpmtest_LIBS)
  string(STRIP "${lpmtest_COMPILER_DEFS}" lpmtest_COMPILER_DEFS)
  string(STRIP "${lpmtest_COMPILER_FLAGS}" lpmtest_COMPILER_FLAGS)
  string(STRIP "${lpmtest_LIBS_DIRS}" lpmtest_LIBS_DIRS)
  string(STRIP "${lpmtest_INCLUDE_DIRS}" lpmtest_INCLUDE_DIRS)

  #---------------------------#
  #   Create the executable   #
  #---------------------------#
  if (lpmtest_LIBS_DIRS)
    link_directories("${lpmtest_LIBS_DIRS}")
  endif()
  add_executable(${target_name} ${target_sources})

  #---------------------------#
  # Set all target properties #
  #---------------------------#
  target_include_directories(${target_name} PUBLIC
        ${PROJECT_BINARY_DIR}
        ${PROJECT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${lpmtest_INCLUDE_DIRS}
        )
  target_link_libraries(${target_name} PUBLIC lpm ${CMAKE_DL_LIBS} ${MPI_C_LIBRARIES} MPI::MPI_C)

 if (NOT lpmtest_EXCLUDE_CATCH_MAIN)
   target_link_libraries(${target_name} PUBLIC lpm_test_main PRIVATE Catch2::Catch2)
 else()
   target_link_libraries(${target_name} PRIVATE Catch2::Catch2WithMain)
 endif()
 if (lpmtest_COMPILER_DEFS)
   target_compile_definitions(${target_name} PUBLIC "${lpmtest_COMPILER_DEFS}")
 endif()
 if (lpmtest_COMPILER_FLAGS)
   target_compile_options(${target_name} PUBLIC "${lpmtest_COMPILER_FLAGS}")
 endif()
 if (lpmtest_LIBS)
   target_link_libraries(${target_name} PUBLIC "${lpmtest_LIBS}")
 endif()
 if (lpmtest_LINKER_FLAGS)
   set_target_properties(${target_name} PROPERTIES LINK_FLAGS "${lpmtest_LINKER_FLAGS}")
 endif()

  #--------------------------#
  # Setup MPI/OpenMP configs #
  #--------------------------#

  list(LENGTH lpmtest_MPI_RANKS NUM_MPI_RANK_ARGS)
  list(LENGTH lpmtest_THREADS   NUM_THREAD_ARGS)

  if (NUM_MPI_RANK_ARGS GREATER 3)
    message(FATAL_ERROR "Too many mpi arguments for ${target_name}")
  endif()
  if (NUM_THREAD_ARGS GREATER 3)
    message(FATAL_ERROR "Too many thread arguments for ${target_name}")
  endif()

  ###### DANGER ##### #### #### #### #### #### #### #### #### #### #### #### ####
  # OpenMPI prohibits running as root by default, for good reasons.  But, if you need
  # to run tests with OpenMPI from a docker image, you have to run as root.
  #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
  set(mpi_allow_root "")
  if ($ENV{DOCKER_ALLOW_MPI_RUN_AS_ROOT})
    if ($ENV{DOCKER_ALLOW_MPI_RUN_AS_ROOT_CONFIRM})
      set(mpi_allow_root "--allow-run-as-root")
    endif()
  endif()

  set(MPI_START_RANK 1)
  set(MPI_END_RANK 1)
  set(MPI_INCREMENT 1)

  set(THREAD_START 1)
  set(THREAD_END 1)
  set(THREAD_INCREMENT 1)

  if (NUM_MPI_RANK_ARGS EQUAL 0)
  elseif(NUM_MPI_RANK_ARGS EQUAL 1)
    list(GET lpmtest_MPI_RANKS 0 RETURN_VAL)
    set(MPI_START_RANK ${RETURN_VAL})
    set(MPI_END_RANK ${RETURN_VAL})
  elseif(NUM_MPI_RANK_ARGS EQUAL 2)
    list(GET lpmtest_MPI_RANKS 0 RETURN_VAL)
    set(MPI_START_RANK ${RETURN_VAL})
    list(GET lpmtest_MPI_RANKS 1 RETURN_VAL)
    set(MPI_END_RANK ${RETURN_VAL})
  else()
    list(GET lpmtest_MPI_RANKS 0 RETURN_VAL)
    set(MPI_START_RANK ${RETURN_VAL})
    list(GET lpmtest_MPI_RANKS 1 RETURN_VAL)
    set(MPI_END_RANK ${RETURN_VAL})
    list(GET lpmtest_MPI_RANKS 2 RETURN_VAL)
    set(MPI_INCREMENT ${RETURN_VAL})
  endif()

  if (NUM_THREAD_ARGS EQUAL 0)
  elseif(NUM_THREAD_ARGS EQUAL 1)
    list(GET lpmtest_THREADS 0 RETURN_VAL)
    set(THREAD_START ${RETURN_VAL})
    set(THREAD_END ${RETURN_VAL})
  elseif(NUM_THREAD_ARGS EQUAL 2)
    list(GET lpmtest_THREADS 0 RETURN_VAL)
    set(THREAD_START ${RETURN_VAL})
    list(GET lpmtest_THREADS 1 RETURN_VAL)
    set(THREAD_END ${RETURN_VAL})
  else()
    list(GET lpmtest_THREADS 0 RETURN_VAL)
    set(THREAD_START ${RETURN_VAL})
    list(GET lpmtest_THREADS 1 RETURN_VAL)
    set(THREAD_END ${RETURN_VAL})
    list(GET lpmtest_THREADS 2 RETURN_VAL)
    set(THREAD_INCREMENT ${RETURN_VAL})
  endif()

  if (NOT MPI_START_RANK GREATER 0)
    message (FATAL_ERROR "Error! MPI_START_RANK is <=0.")
  endif()
  if (NOT MPI_END_RANK GREATER 0)
    message (FATAL_ERROR "Error! MPI_END_RANK is <=0.")
  endif()
  if (MPI_INCREMENT GREATER 0 AND MPI_START_RANK GREATER MPI_END_RANK)
    message (FATAL_ERROR "Error! MPI_START_RANK > MPI_END_RANK, but the increment is positive.")
  endif()
  if (MPI_INCREMENT LESS 0 AND MPI_START_RANK LESS MPI_END_RANK)
    message (FATAL_ERROR "Error! MPI_START_RANK < MPI_END_RANK, but the increment is negative.")
  endif()
  if (NOT THREAD_START GREATER 0)
    message (FATAL_ERROR "Error! THREAD_START is <=0.")
  endif()
  if (NOT THREAD_END GREATER 0)
    message (FATAL_ERROR "Error! THREAD_END is <=0.")
  endif()
  if (THREAD_INCREMENT GREATER 0 AND THREAD_START GREATER THREAD_END)
    message (FATAL_ERROR "Error! THREAD_START > THREAD_END, but the increment is positive.")
  endif()
  if (THREAD_INCREMENT LESS 0 AND THREAD_START LESS THREAD_END)
    message (FATAL_ERROR "Error! THREAD_START < THREAD_END, but the increment is negative.")
  endif()

  # Check both, in case user has negative increment
  if (MPI_END_RANK GREATER 1 OR MPI_START_RANK GREATER 1)
    if ("${lpmtest_MPI_EXEC_NAME}" STREQUAL "")
      set (lpmtest_MPI_EXEC_NAME "mpiexec")
    endif()
    if ("${lpmtest_MPI_NP_FLAG}" STREQUAL "")
      set (lpmtest_MPI_NP_FLAG "${mpi_allow_root} -n")
    endif()
  endif()

  #------------------------------------------------#
  # Loop over MPI/OpenMP configs, and create tests #
  #------------------------------------------------#

  if (lpmtest_EXE_ARGS)
    set(invokeExec "./${target_name} ${lpmtest_EXE_ARGS}")
  else()
    set(invokeExec "./${target_name}")
  endif()

  foreach (NRANKS RANGE ${MPI_START_RANK} ${MPI_END_RANK} ${MPI_INCREMENT})
    foreach (NTHREADS RANGE ${THREAD_START} ${THREAD_END} ${THREAD_INCREMENT})
      # Create the test name
      set(FULL_TEST_NAME ${target_name}_ut_np${NRANKS}_omp${NTHREADS})

      set(USE_MPI FALSE)
      if (${NRANKS} GREATER 1)
        set(USE_MPI TRUE)
      endif()

      # Create the test
      if (USE_MPI)
        add_test(NAME ${FULL_TEST_NAME}
                 COMMAND sh -c "${lpmtest_MPI_EXEC_NAME} ${lpmtest_MPI_NP_FLAG} ${NRANKS} ${lpmtest_MPI_EXTRA_ARGS} ${invokeExec}")
      else()
        add_test(NAME ${FULL_TEST_NAME}
                 COMMAND sh -c "${invokeExec}")
      endif()

      # Set test properties
      math(EXPR CURR_CORES "${NRANKS}*${NTHREADS}")
      set_tests_properties(${FULL_TEST_NAME} PROPERTIES ENVIRONMENT OMP_NUM_THREADS=${NTHREADS} PROCESSORS ${CURR_CORES} PROCESSOR_AFFINITY True)
      if (lpmtest_DEP AND NOT lpmtest_DEP STREQUAL "${FULL_TEST_NAME}")
        set_tests_properties(${FULL_TEST_NAME} PROPERTIES DEPENDS ${lpmtest_DEP})
      endif()

      if (lpmtest_LABELS)
        set_tests_properties(${FULL_TEST_NAME} PROPERTIES LABELS "${lpmtest_LABELS}")
      endif()

      if (lpmtest_PROPERTIES)
        set_tests_properties(${FULL_TEST_NAME} PROPERTIES ${lpmtest_PROPERTIES})
      endif()

      set (RES_GROUPS "devices:1")
      if (USE_MPI)
        foreach (rank RANGE 2 ${NRANKS})
          set (RES_GROUPS "${RES_GROUPS},devices:1")
        endforeach()
      endif()
      set_property(TEST ${FULL_TEST_NAME} PROPERTY RESOURCE_GROUPS "${RES_GROUPS}")
    endforeach()
  endforeach()

  if (lpmtest_SERIAL)
    # All tests run serially
    set (tests_names)
    foreach (NRANKS RANGE ${MPI_START_RANK} ${MPI_END_RANK} ${MPI_INCREMENT})
      foreach (NTHREADS RANGE ${THREAD_START} ${THREAD_END} ${THREAD_INCREMENT})
        # Create the test
        set(FULL_TEST_NAME ${target_name}_ut_np${NRANKS}_omp${NTHREADS})
        list(APPEND tests_names ${FULL_TEST_NAME})
      endforeach ()
    endforeach()
    set_tests_properties (${tests_names} PROPERTIES RESOURCE_LOCK ${target_name}_serial)
  endif ()

endfunction()
