include(CMakeDependentOption)

 
set(CMAKE_CUDA_ARCHITECTURES OFF)

# if possible, by default we only build locally for the native host arch to save build times and binaries size CMake customizations
# and function definitions
set(CUDA_ARCH "native" CACHE STRING "Cuda architecture to build")

option(CUDA_ARCH "Build for cuda host architecture only" "native")
# build archs controlled by cmake options must by either native, all OR at least one of these(ampere|ada|hopper|blackwell)
 
function(set_target_cuda_arch_flags TARGET_NAME)        
    set_target_properties( ${TARGET_NAME} PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}")              
endfunction()

