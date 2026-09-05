include(cmake/libs/hip/debug.cmake)
include(cmake/libs/hip/target_generation.cmake)

find_package(HIP CONFIG REQUIRED)
option(ENABLE_HIP_LINE_INFO "Enable line info for HIP kernels compilation" ON)

function(add_hip_to_target TARGET_NAME COMPONENTS)
    set_default_hip_target_properties(${TARGET_NAME})

    if (${ENABLE_DEBUG})
        add_hip_debug_support_to_target(${TARGET_NAME})
    endif()
    if (${ENABLE_HIP_LINE_INFO})
        add_hip_lineinfo_to_target(${TARGET_NAME})
    endif()
   # target_link_libraries(${TARGET_NAME} PRIVATE hip::device)

endfunction()
