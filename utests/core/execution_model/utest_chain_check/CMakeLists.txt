# Tests of the check of the chain that the executors of TransformDPP make
# (include/fused_kernel/core/execution_model/operation_model/chain_check.h).
#
# utest_chain_check.h is discovered like any other test (the predicate on the types of the chains, and the correct
# chains executed, on the target of every enabled backend: utest_chain_check_cpp, _cu, _hip). This file adds:
#  - the same test with the executor of ParArch::CPU_MULTITHREAD (utest_chain_check_cpu_mt_cpp), when it is enabled
#  - the compile-fail tests. For each of the wrong chains and each backend there is a target built from the same
#    header with -DFKL_CHAIN_CHECK_WRONG=<chain>, which is NOT part of "all", and a ctest test that builds only that
#    target through check_rejected.cmake. The test passes only if the build fails, its output has the message of the
#    check and it has no other error: a wrong chain that compiles, that stops with another error, or that is
#    instantiated after the report (a second error) is a failed test.
#    The tests share a resource lock because they all run the build tool on the same build directory.

set(CHAIN_CHECK_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/utest_chain_check.h")
set(CHAIN_CHECK_MESSAGE_1 "FKL: the OutputType of the previous Op is not convertible to the InputType of the next Op")
set(CHAIN_CHECK_MESSAGE_2 "FKL: Complete Read Operations as continuations are not allowed")
set(CHAIN_CHECK_MESSAGE_3 "FKL: Fusing two batch operations of different BATCH size is not allowed")
set(CHAIN_CHECK_MESSAGE_4 "${CHAIN_CHECK_MESSAGE_1}") # a type mismatch after a MidWrite
set(CHAIN_CHECK_WRONG_CHAINS 1 2 3 4)

# An executable built from utest_chain_check.h for the backend of EXTENSION (cpp, cu or hip), with the compile
# definitions DEFINITIONS. With EXCLUDE it is not part of "all" and has no test that runs it.
function(add_chain_check_target TARGET_NAME_EXT EXTENSION DEFINITIONS EXCLUDE)
    set(TEST_GENERATED_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME_EXT}/launcher.${EXTENSION}")
    set(TEST_SOURCE "${CHAIN_CHECK_SOURCE}")
    configure_file(${CMAKE_SOURCE_DIR}/tests/launcher.in ${TEST_GENERATED_SOURCE} @ONLY)
    if (EXCLUDE)
        add_executable(${TARGET_NAME_EXT} EXCLUDE_FROM_ALL "${TEST_GENERATED_SOURCE};${CHAIN_CHECK_SOURCE}")
    else()
        add_executable(${TARGET_NAME_EXT} "${TEST_GENERATED_SOURCE};${CHAIN_CHECK_SOURCE}")
    endif()
    configure_test_target_flags("${TARGET_NAME_EXT}" "${CHAIN_CHECK_SOURCE}" "")
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

# The wrong chains of one backend: a target each, and a test that passes if the build of the target fails with the
# message of the check and no other error. TAG names the backend in the targets (cpp, cpu_mt, cu, hip)
function(add_chain_check_rejections TAG EXTENSION DEFINITIONS)
    foreach(CHAIN ${CHAIN_CHECK_WRONG_CHAINS})
        set(TARGET_NAME_EXT "utest_chain_check_wrong${CHAIN}_${TAG}")
        set(WRONG_DEFINITIONS ${DEFINITIONS} FKL_CHAIN_CHECK_WRONG=${CHAIN})
        add_chain_check_target("${TARGET_NAME_EXT}" "${EXTENSION}" "${WRONG_DEFINITIONS}" TRUE)
        add_test(NAME ${TARGET_NAME_EXT}_rejected
                 COMMAND ${CMAKE_COMMAND} -DBUILD_DIR=${CMAKE_BINARY_DIR} -DTARGET=${TARGET_NAME_EXT}
                         -DCONFIG=$<CONFIG> "-DMESSAGE=${CHAIN_CHECK_MESSAGE_${CHAIN}}"
                         -P ${CMAKE_CURRENT_SOURCE_DIR}/check_rejected.cmake)
        set_tests_properties(${TARGET_NAME_EXT}_rejected PROPERTIES RESOURCE_LOCK fkl_chain_check_build)
    endforeach()
endfunction()

if (${ENABLE_CPU})
    add_chain_check_rejections(cpp cpp "")
    if (${ENABLE_CPU_MULTITHREAD})
        add_chain_check_target(utest_chain_check_cpu_mt_cpp cpp "FKL_CHAIN_CHECK_CPU_MT" FALSE)
        add_chain_check_rejections(cpu_mt cpp "FKL_CHAIN_CHECK_CPU_MT")
    endif()
endif()

if (CMAKE_CUDA_COMPILER AND ENABLE_CUDA)
    add_chain_check_rejections(cu cu "")
endif()

if (CMAKE_HIP_COMPILER AND ENABLE_HIP)
    add_chain_check_rejections(hip hip "")
endif()
