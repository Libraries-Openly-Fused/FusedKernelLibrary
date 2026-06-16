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
set(CUDA_ARCH "native" CACHE STRING "Cuda architecture to build")

option(CUDA_ARCH "Build for cuda host architecture only" "native")
# build archs controlled by cmake options must by either native, all OR at least one of these(turing|ampere|ada|hopper|)
#for cuda <13 we need to avoid < 7.0 compute capabilities 
 

function(set_target_cuda_arch_flags TARGET_NAME)        
    set_target_properties( ${TARGET_NAME} PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}")         
     
endfunction()

