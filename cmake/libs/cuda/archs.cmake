include(CMakeDependentOption)

function (remove_pre70gpus GPU_ARCHS GPU_MINUM70)
    set(GPU_MIN "")
  
    foreach(GPU_ARCH IN LISTS GPU_ARCHS)    
     
        if (GPU_ARCH LESS 70)
            continue()
        else()
         
            list(APPEND GPU_MIN ${GPU_ARCH})
        endif()
    endforeach()  
      set(GPU_MINUM70 ${GPU_MIN} PARENT_SCOPE)  

    
endfunction()

set(CMAKE_CUDA_ARCHITECTURES OFF)


# if possible, by default we only build locally for the native host arch to save build times and binaries size CMake customizations
# and function definitions


if(${CMAKE_VERSION} GREATER_EQUAL "3.24.0")
    set(CUDA_ARCH "native" CACHE STRING "Cuda architecture to build")
else()
    #default build for all known builds with old cmake (ubuntu 22.04 and jetpack 6.2)
    set(CUDA_ARCH "all" CACHE STRING "Cuda architecture to build")
endif()


option(CUDA_ARCH "Build for cuda host architecture only" "native")
# build archs controlled by cmake options must by either native, all OR at least one of these(turing|ampere|ada|hopper|)
#for cuda <13 we need to avoid < 7.0 compute capabilities 
if (${CUDA_VERSION_MAJOR} LESS 13)
    set (GPU_MINUM70 "")  
    #nvcc automatically builds all gpu arch with all, so we need to remove <7.0
    if ("${CUDA_ARCH}" STREQUAL "all" OR "${CUDA_ARCH}" STREQUAL "all-major")  
        execute_process(COMMAND "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" "--list-gpu-arch" 
        OUTPUT_VARIABLE GPU_ARCHS OUTPUT_STRIP_TRAILING_WHITESPACE)
        
        string(REPLACE "compute_" "" GPU_ARCHS ${GPU_ARCHS})
        string(REPLACE "\n" ";" GPU_ARCHS ${GPU_ARCHS})
    
        list (FIND GPU_ARCHS "61" _index)
        if (${_index} GREATER -1)
            message(WARNING "Skipping building for deprecated GPU architectures older than 70")
        endif()
        remove_pre70gpus("${GPU_ARCHS}" "${GPU_MINUM70}") 
        set(CUDA_ARCH "${GPU_MINUM70}" CACHE STRING "Cuda architecture to build" FORCE)
    endif()  
    if("${CUDA_ARCH}" STREQUAL "native")
        execute_process(COMMAND "nvidia-smi" "--query-gpu=compute_cap" "--format=noheader" 
        OUTPUT_VARIABLE GPU_CC OUTPUT_STRIP_TRAILING_WHITESPACE)
        string(REPLACE "." "" GPU_CC ${GPU_CC})
        message(STATUS "Detected native GPU architecture: ${GPU_CC}")
        remove_pre70gpus("${GPU_CC}" "${GPU_MINUM70}")
        list(LENGTH GPU_MINUM70 length)
        if(${length} EQUAL 0)
            message(ERROR "No valid CUDA architecture to build. only 70 and above are supported")
        endif()
        set(CUDA_ARCH "native" CACHE STRING "Cuda architecture to build" FORCE)
    endif()
    list(LENGTH CUDA_ARCH length)

    if(${length} GREATER 0)
        #check that we are not manually specifung an old gpu arch to build
        remove_pre70gpus("${CUDA_ARCH}" "${GPU_MINUM70}")
        list(LENGTH GPU_MINUM70 length1)
        if(${length1} EQUAL 0)
            message(ERROR "No valid CUDA architecture to build. only 70 and above are supported")
            endif()
        endif()
        set(CUDA_ARCH "${GPU_MINUM70}" CACHE STRING "Cuda architecture to build" FORCE)
    endif()

function(set_target_cuda_arch_flags TARGET_NAME)        
    set_target_properties( ${TARGET_NAME} PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}")         
     
endfunction()

