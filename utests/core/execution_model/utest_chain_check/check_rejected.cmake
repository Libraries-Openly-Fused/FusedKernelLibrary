# Builds one target that must be rejected by the check of the chain (include/.../operation_model/chain_check.h), and
# checks that the build fails, that its output has the message of the check, and that it has no other error: a chain
# that is instantiated after the report, or that stops with a different error as well, is a failed test.
# usage: cmake -DBUILD_DIR=<build dir> -DTARGET=<target> -DCONFIG=<configuration or empty> -DMESSAGE=<message> -P check_rejected.cmake
set(build_args --build "${BUILD_DIR}" --target "${TARGET}")
if (CONFIG)
    list(APPEND build_args --config "${CONFIG}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} ${build_args}
                OUTPUT_VARIABLE output ERROR_VARIABLE error_output RESULT_VARIABLE result)
set(all_output "${output}\n${error_output}")
# The template arguments in the diagnostics contain ';', which would split the list of lines below
string(REPLACE ";" "<semicolon>" all_output "${all_output}")

if (result EQUAL 0)
    message(FATAL_ERROR "${TARGET}: the wrong chain compiled")
endif()
if (NOT all_output MATCHES "${MESSAGE}")
    message(FATAL_ERROR "${TARGET}: the build failed without the message of the check \"${MESSAGE}\":\n${all_output}")
endif()

# The lines that report an error: "file:line:col: error: ..." (g++, clang++, amdclang++), "file(line): error: ..."
# (nvcc) and "file(line): error C2338: ..." (MSVC)
string(REGEX MATCHALL "[^\n]*: error[: ][^\n]*" error_lines "${all_output}")
list(FILTER error_lines EXCLUDE REGEX "${MESSAGE}")
list(LENGTH error_lines other_errors)
if (other_errors GREATER 0)
    string(REPLACE ";" "\n" error_lines "${error_lines}")
    message(FATAL_ERROR "${TARGET}: ${other_errors} error(s) other than the message of the check:\n${error_lines}")
endif()
message(STATUS "${TARGET}: rejected with the message of the check and no other error")
