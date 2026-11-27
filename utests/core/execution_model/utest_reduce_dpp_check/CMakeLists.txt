# Tests of the check of the IOps that the executors of ReduceDPP make
# (include/fused_kernel/core/execution_model/reduce_dpp.h), made as the ones of utest_chain_check.
#
# utest_reduce_dpp_check.h is discovered like any other test (the correct execution, on the target of every enabled
# backend: utest_reduce_dpp_check_cpp, _cu, _hip). This file adds:
#  - the same test with the executor of ParArch::CPU_MULTITHREAD (utest_reduce_dpp_check_cpu_mt_cpp), when it is
#    enabled
#  - the compile-fail tests. For each of the wrong executions and each backend there is a target built from the same
#    header with -DFKL_REDUCE_CHECK_WRONG=<execution>, which is NOT part of "all", and a ctest test that builds only
#    that target through check_rejected.cmake of utest_chain_check. The test passes only if the build fails, its output
#    has the message of the check and it has no other error: a wrong execution that compiles, that stops with another
#    error, or that is instantiated after the report (a second error) is a failed test. g++ does not give that second
#    error (it does not instantiate the function templates that an instantiation uses after its first error): for it,
#    and for every other compiler, the correct variant of utest_reduce_dpp_check.h also executes wrong IOps whose
#    report says nothing (ReduceCheckNotAnIOp), which compiles only if the executor instantiates nothing else.
#    The tests share the resource lock of the ones of utest_chain_check, because they all run the build tool on the
#    same build directory.

set(REDUCE_CHECK_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/utest_reduce_dpp_check.h")
set(REDUCE_CHECK_REJECTED "${CMAKE_CURRENT_SOURCE_DIR}/../utest_chain_check/check_rejected.cmake")
set(REDUCE_CHECK_MESSAGE_1 "ReduceDPP needs compute IOps")
set(REDUCE_CHECK_MESSAGE_2 "ReduceDPP: the OutputType of a FirstIOp must be the accumulator type")
set(REDUCE_CHECK_MESSAGE_3 "ReduceDPP needs its reductions as a Tuple of one or more Tuple")
set(REDUCE_CHECK_MESSAGE_4 "ReduceDPP: the OutputType of a FinalIOp is not convertible to the InputType of the write")
set(REDUCE_CHECK_MESSAGE_5 "ReduceDPP: the accumulator type, the OutputType of an AccumulationIOp, must be trivially")
set(REDUCE_CHECK_MESSAGE_6 "ReduceDPP needs a complete Read or ReadBack IOp to read from")
set(REDUCE_CHECK_MESSAGE_7 "ReduceDPP: the OutputType of the read is not convertible to the InputType of a FirstIOp")
set(REDUCE_CHECK_MESSAGE_8 "ReduceDPP: an AccumulationIOp must take a Tuple of two values of its OutputType")
set(REDUCE_CHECK_MESSAGE_9 "ReduceDPP: the accumulator type, the OutputType of an AccumulationIOp, is not convertible")
set(REDUCE_CHECK_MESSAGE_10 "ReduceDPP needs a Write IOp to write the results with")
set(REDUCE_CHECK_WRONG_EXECUTIONS 1 2 3 4 5 6 7 8 9 10)

# An executable built from utest_reduce_dpp_check.h for the backend of EXTENSION (cpp, cu or hip), with the compile
# definitions DEFINITIONS. With EXCLUDE it is not part of "all" and has no test that runs it.
function(add_reduce_check_target TARGET_NAME_EXT EXTENSION DEFINITIONS EXCLUDE)
    set(TEST_GENERATED_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME_EXT}/launcher.${EXTENSION}")
    set(TEST_SOURCE "${REDUCE_CHECK_SOURCE}")
    configure_file(${CMAKE_SOURCE_DIR}/tests/launcher.in ${TEST_GENERATED_SOURCE} @ONLY)
    if (EXCLUDE)
        add_executable(${TARGET_NAME_EXT} EXCLUDE_FROM_ALL "${TEST_GENERATED_SOURCE};${REDUCE_CHECK_SOURCE}")
    else()
        add_executable(${TARGET_NAME_EXT} "${TEST_GENERATED_SOURCE};${REDUCE_CHECK_SOURCE}")
    endif()
    configure_test_target_flags("${TARGET_NAME_EXT}" "${REDUCE_CHECK_SOURCE}" "")
    target_sources(${TARGET_NAME_EXT} PRIVATE ${LAUNCH_SOURCES})
    if (DEFINITIONS)
        target_compile_definitions(${TARGET_NAME_EXT} PRIVATE ${DEFINITIONS})
    endif()
    if (${EXTENSION} STREQUAL "cu")
        add_cuda_to_test(${TARGET_NAME_EXT})
    elseif (${EXTENSION} STREQUAL "hip")
        add_hip_to_test(${TARGET_NAME_EXT})
    endif()
    set_ide_target_folder(${TARGET_NAME_EXT} "/utests/core/execution_model" ${EXTENSION})
    if (NOT EXCLUDE)
        add_test(NAME ${TARGET_NAME_EXT} COMMAND ${TARGET_NAME_EXT})
    endif()
endfunction()

# The wrong executions of one backend: a target each, and a test that passes if the build of the target fails with the
# message of the check and no other error. TAG names the backend in the targets (cpp, cpu_mt, cu, hip)
function(add_reduce_check_rejections TAG EXTENSION DEFINITIONS)
    foreach(EXECUTION ${REDUCE_CHECK_WRONG_EXECUTIONS})
        set(TARGET_NAME_EXT "utest_reduce_dpp_check_wrong${EXECUTION}_${TAG}")
        set(WRONG_DEFINITIONS ${DEFINITIONS} FKL_REDUCE_CHECK_WRONG=${EXECUTION})
        add_reduce_check_target("${TARGET_NAME_EXT}" "${EXTENSION}" "${WRONG_DEFINITIONS}" TRUE)
        add_test(NAME ${TARGET_NAME_EXT}_rejected
                 COMMAND ${CMAKE_COMMAND} -DBUILD_DIR=${CMAKE_BINARY_DIR} -DTARGET=${TARGET_NAME_EXT}
                         -DCONFIG=$<CONFIG> "-DMESSAGE=${REDUCE_CHECK_MESSAGE_${EXECUTION}}"
                         -P ${REDUCE_CHECK_REJECTED})
        set_tests_properties(${TARGET_NAME_EXT}_rejected PROPERTIES RESOURCE_LOCK fkl_chain_check_build)
    endforeach()
endfunction()

if (${ENABLE_CPU})
    add_reduce_check_rejections(cpp cpp "")
    if (${ENABLE_CPU_MULTITHREAD})
        add_reduce_check_target(utest_reduce_dpp_check_cpu_mt_cpp cpp "FKL_REDUCE_CHECK_CPU_MT" FALSE)
        add_reduce_check_rejections(cpu_mt cpp "FKL_REDUCE_CHECK_CPU_MT")
    endif()
endif()

if (CMAKE_CUDA_COMPILER AND ENABLE_CUDA)
    add_reduce_check_rejections(cu cu "")
endif()

if (CMAKE_HIP_COMPILER AND ENABLE_HIP)
    add_reduce_check_rejections(hip hip "")
endif()
