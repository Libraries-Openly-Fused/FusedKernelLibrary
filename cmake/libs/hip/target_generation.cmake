function(set_default_hip_target_properties TARGET_NAME)
    if (WIN32)
        target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-Xcompiler=/bigobj /Zc:preprocessor>)
    endif()
    set_target_properties(${TARGET_NAME} PROPERTIES
        HIP_STANDARD 17
        HIP_STANDARD_REQUIRED ON
        HIP_EXTENSIONS OFF)
    if (NOT(${TEMPLATE_DEPTH} STREQUAL "default"))
        target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-ftemplate-depth=${TEMPLATE_DEPTH}>)
        if (NOT WIN32)
            target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ftemplate-depth=${TEMPLATE_DEPTH}>)
        endif()
    endif()
endfunction()

function(add_hip_debug_support_to_target TARGET_NAME)
    target_compile_options(${TARGET_NAME} PRIVATE $<$<AND:$<CONFIG:debug>,$<COMPILE_LANGUAGE:HIP>>:-ggdb>)
endfunction()

function(add_hip_lineinfo_to_target TARGET_NAME)
    if (NOT ${ENABLE_HIP_DEBUG})
        target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-gline-tables-only>)
    endif()
endfunction()
