set(CMAKE_HIP_ARCHITECTURES OFF)

set(HIP_ARCH "gfx1200" CACHE STRING "HIP architecture(s) to build")

function(set_target_hip_arch_flags TARGET_NAME)
    set_target_properties(${TARGET_NAME} PROPERTIES HIP_ARCHITECTURES "${HIP_ARCH}")
endfunction()
