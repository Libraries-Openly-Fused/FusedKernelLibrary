function(set_default_hip_target_properties TARGET_NAME)
    # LTS-C++17: HIP sources are compiled as strict C++17, like the CXX and CUDA sources
    set_target_properties(${TARGET_NAME} PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON HIP_EXTENSIONS OFF
                                                    HIP_RUNTIME_LIBRARY Shared)

    if (NOT(${TEMPLATE_DEPTH} STREQUAL "default"))
        target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-ftemplate-depth=${TEMPLATE_DEPTH}>)
    endif()
endfunction()
