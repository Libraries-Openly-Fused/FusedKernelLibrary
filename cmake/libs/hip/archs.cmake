# HIP GPU architecture selection
# Defaults to "native" which lets the compiler auto-detect the target GPU
set(HIP_ARCH "native" CACHE STRING "HIP/ROCm GPU architecture to build for (e.g. native, gfx1100, gfx90a)")

function(set_target_hip_arch_flags TARGET_NAME)
    if ("${HIP_ARCH}" STREQUAL "native")
        set_target_properties(${TARGET_NAME} PROPERTIES HIP_ARCHITECTURES "native")
    else()
        set_target_properties(${TARGET_NAME} PROPERTIES HIP_ARCHITECTURES "${HIP_ARCH}")
    endif()
endfunction()
