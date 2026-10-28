include(cmake/libs/hip/debug.cmake)
include(cmake/libs/hip/target_generation.cmake)

# hip::host gives the HIP runtime (amdhip64) to targets whose C++ sources are compiled by the host compiler (e.g.
# tests/main.cpp). hip::device is not used: it adds "-x hip --offload-arch=..." to the CXX sources of a target, which
# breaks when the CXX compiler is not amdclang++. The .hip sources get their offload flags from HIP_ARCHITECTURES.
find_package(hip CONFIG REQUIRED)
option(ENABLE_HIP_LINE_INFO "Enable line info for HIP kernels compilation" ON)

function(add_hip_to_target TARGET_NAME COMPONENTS)
    set_default_hip_target_properties(${TARGET_NAME})
    set_target_hip_arch_flags(${TARGET_NAME})

    if (ENABLE_DEBUG)
        add_hip_debug_support_to_target(${TARGET_NAME})
    endif()
    if (ENABLE_HIP_LINE_INFO)
        add_hip_lineinfo_to_target(${TARGET_NAME})
    endif()
    target_link_libraries(${TARGET_NAME} PRIVATE hip::host)
endfunction()
