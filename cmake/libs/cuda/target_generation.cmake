function(set_default_cuda_target_properties TARGET_NAME)
    if (WIN32)
        list(APPEND COMPILER_CUDA_FLAGS -Xcompiler=/bigobj)
    endif()
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${COMPILER_CUDA_FLAGS}>)
	if(CUDA_VERSION_MAJOR EQUAL 13)	
    set_target_properties(${TARGET_NAME} PROPERTIES CUDA_STANDARD_REQUIRED ON CUDA_STANDARD 17 CUDA_RUNTIME_LIBRARY
                                                                                               Hybrid)
    else()
    set_target_properties(${TARGET_NAME} PROPERTIES CUDA_STANDARD_REQUIRED ON CUDA_STANDARD 17 CUDA_RUNTIME_LIBRARY
                                                                                               Shared)
    endif()
    
    set_target_cuda_arch_flags(${TARGET_NAME})
    
    # use less precise but faster cuda math methods
    #target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
    # parallel compilation of cuda kernels
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--threads 0>)
    #disable relocatable device code   
    #see https://forums.developer.nvidia.com/t/the-cost-of-relocatable-device-code-rdc-true/47665
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=false>)
    
    #cuda 12 can compile in parallel, so let's use this 
    if (${CUDA_VERSION_MAJOR} GREATER_EQUAL 12)
        #split compile does not work with gpu debug code
        if(NOT ${ENABLE_DEBUG})
            target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-split-compile=0>)
        endif()
    endif()
    
    if (NOT(${TEMPLATE_DEPTH} STREQUAL  "default"))
    #bugfix for windows compilation of tests with more than 200 recursions
    #set it in device code and host
        target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-ftemplate-depth=${TEMPLATE_DEPTH}>)

        if (WIN32)
       #cannot be set, no compiler flag available
        else() 
        #asume clang or gcc on linux
        target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ftemplate-depth=${TEMPLATE_DEPTH}>)           
        endif()
    endif()
endfunction()
