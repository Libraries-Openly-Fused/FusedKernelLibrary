option(ENABLE_LINE_INFO "Enable line info for HIP kernels compilation" ON)
option(ENABLE_HIP_DEBUG "Generate HIP debug information for device code" OFF)

include(cmake/libs/hip/target_generation.cmake)

find_package(hip REQUIRED)

function(add_hip_to_target TARGET_NAME)
    set_default_hip_target_properties(${TARGET_NAME})
    set_target_hip_arch_flags(${TARGET_NAME})

    if (${ENABLE_HIP_DEBUG})
        add_hip_debug_support_to_target(${TARGET_NAME})
    endif()
    if (${ENABLE_LINE_INFO})
        add_hip_lineinfo_to_target(${TARGET_NAME})
    endif()
    target_link_libraries(${TARGET_NAME} PRIVATE hip::host)
endfunction()
