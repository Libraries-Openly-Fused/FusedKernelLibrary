cmake_policy(SET CMP0104 NEW) # Initialize CMAKE_CUDA_ARCHITECTURES when CMAKE_CUDA_COMPILER_ID is NVIDIA

check_language(CUDA)
 
if (NOT CMAKE_CUDA_COMPILER)    
    message(WARNING "CUDA compiler not found. Disabling CUDA support.")
    set(ENABLE_CUDA OFF CACHE BOOL "Enable CUDA support" FORCE)
    return()    
endif()

enable_language(CUDA)   

include (cmake/libs/cuda/cuda.cmake) 
include (cmake/libs/cuda/archs.cmake)

if(UNIX)
    return()
endif()

# for ninja in CI we will always use the NVCC path

#if(CMAKE_GENERATOR MATCHES "Ninja")
#    set(CMAKE_CUDA_COMPILER ${CUDA_PATH}/nvcc/bin/nvcc.exe)
#    set(CUDAToolkit_ROOT ${CUDA_PATH}/nvcc)
#endif()
#if(CMAKE_GENERATOR MATCHES "Visual Studio")
 #   set(CUDAToolkit_ROOT ${CUDA_PATH}/nvcc)
#endif()
