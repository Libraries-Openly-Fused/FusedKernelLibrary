option(ENABLE_NVTX "Enable NVTX utilities" OFF)
option(ENABLE_DEBUG "Generate CUDA debug information for device code and turn off all device code optimizations"
       OFF)

function(add_cuda_debug_support_to_target TARGET_NAME)
    # optional debug code in gpu ()    
    target_compile_options(${TARGET_NAME} PRIVATE $<$<AND:$<CONFIG:debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>)    
endfunction()

function(add_cuda_lineinfo_to_target TARGET_NAME)
    #-G already implies lineinfo, so only pass flag if debug code is not 
    if (NOT ${ENABLE_DEBUG})
        if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
            target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -lineinfo>)    
        endif()
    endif()    
endfunction()


function(add_nvtx_support_to_target TARGET_NAME)    
    if (${CMAKE_VERSION} GREATER_EQUAL "3.25.0")
        target_link_libraries(${TARGET_NAME} PRIVATE CUDA::nvtx3)
        else()
        target_include_directories(${TARGET_NAME} PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/include)
    endif()
        
    target_compile_definitions(${TARGET_NAME} PUBLIC ENABLE_NVTX)    
endfunction()
