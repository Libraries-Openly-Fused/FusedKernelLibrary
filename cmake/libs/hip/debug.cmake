# Also declared by cmake/libs/cuda/debug.cmake: a HIP only configure needs it too
option(ENABLE_DEBUG "Generate debug information for GPU device code" OFF)

function(add_hip_debug_support_to_target TARGET_NAME)
    target_compile_options(${TARGET_NAME} PRIVATE $<$<AND:$<CONFIG:debug>,$<COMPILE_LANGUAGE:HIP>>:-g>)
endfunction()

function(add_hip_lineinfo_to_target TARGET_NAME)
    # clang keeps the last -g* flag, so -gline-tables-only would replace the full debug information of -g: only pass
    # it outside the Debug configuration, the only one that gets -g (from CMAKE_HIP_FLAGS_DEBUG and ENABLE_DEBUG)
    target_compile_options(${TARGET_NAME} PRIVATE
                           $<$<AND:$<NOT:$<CONFIG:debug>>,$<COMPILE_LANGUAGE:HIP>>:-gline-tables-only>)
endfunction()
