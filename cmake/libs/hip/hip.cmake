option(ENABLE_HIP_LINE_INFO "Enable line info for HIP kernels compilation" ON)
option(ENABLE_HIP_DEBUG "Generate HIP debug information for device code" OFF)
 
include(cmake/libs/hip/target_generation.cmake)
set(ROCM_ROOT "/opt/rocm-7.2.0" CACHE PATH "Root directory of the ROCm installation")
list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")

find_package(hip CONFIG REQUIRED)

function(add_hip_to_target TARGET_NAME)
    set_default_hip_target_properties(${TARGET_NAME})
    set_target_hip_arch_flags(${TARGET_NAME})

    if (${ENABLE_HIP_DEBUG})
        add_hip_debug_support_to_target(${TARGET_NAME})
    endif()
    if (${ENABLE_HIP_LINE_INFO})
        add_hip_lineinfo_to_target(${TARGET_NAME})
    endif()
     #hip-lang::device hip-lang::amdhip64
    target_link_libraries(${TARGET_NAME} PRIVATE hip::host)
endfunction()
