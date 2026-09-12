function(add_hip_debug_support_to_target TARGET_NAME)
    target_compile_options(${TARGET_NAME} PRIVATE $<$<AND:$<CONFIG:debug>,$<COMPILE_LANGUAGE:HIP>>:-g>)
endfunction()

function(add_hip_lineinfo_to_target TARGET_NAME)
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-gline-tables-only>)
endfunction()
