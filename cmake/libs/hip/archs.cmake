# HIP_ARCH and the CMAKE_HIP_ARCHITECTURES it seeds are set in hip_init.cmake, before enable_language(HIP).
function(set_target_hip_arch_flags TARGET_NAME)
    set_target_properties(${TARGET_NAME} PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
endfunction()
