function(set_default_hip_target_properties TARGET_NAME)
    set_target_properties(${TARGET_NAME} PROPERTIES HIP_STANDARD_REQUIRED ON HIP_STANDARD 20 HIP_RUNTIME_LIBRARY Shared)

    set_target_hip_arch_flags(${TARGET_NAME})

    if (NOT(${TEMPLATE_DEPTH} STREQUAL "default"))
        target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-ftemplate-depth=${TEMPLATE_DEPTH}>)
    endif()
endfunction()
