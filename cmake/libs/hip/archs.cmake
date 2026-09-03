set(HIP_ARCH "gfx1200" CACHE STRING "HIP architecture(s) to build")

function(set_target_hip_arch_flags TARGET_NAME)
    if(DEFINED CMAKE_HIP_ARCHITECTURES)
        set_target_properties(${TARGET_NAME} PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
    else()
        set_target_properties(${TARGET_NAME} PROPERTIES HIP_ARCHITECTURES "${HIP_ARCH}")
    endif()
endfunction()
