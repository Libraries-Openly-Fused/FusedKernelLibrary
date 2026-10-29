# HIP (ROCm) backend: FKL's GPU_AMD ParArch, compiled by amdclang++ through CMake's HIP language.
# HIP support is optional: CMakeLists.txt includes this file only if check_language(HIP) found a HIP compiler for AMD
# GPUs and ENABLE_HIP is ON.

# By default enable_language(HIP) sets CMAKE_HIP_ARCHITECTURES to the GPU(s) of this machine (or the compiler default).
# -DHIP_ARCH=<list> (e.g. "gfx1200;gfx1201") seeds it instead on the first configure of a build directory; an explicit
# -DCMAKE_HIP_ARCHITECTURES wins over HIP_ARCH.
if (DEFINED HIP_ARCH AND NOT DEFINED CACHE{CMAKE_HIP_ARCHITECTURES})
    set(CMAKE_HIP_ARCHITECTURES "${HIP_ARCH}" CACHE STRING "HIP architectures")
endif()
enable_language(HIP)
# Where find_package(hip) looks for the HIP package: first the ROCm installation of the HIP compiler that CMake found,
# so that hip::host links the HIP runtime of that ROCm (the one the HIP language links), then $HIP_PATH / $ROCM_PATH
foreach(ROCM_DIR "${CMAKE_HIP_COMPILER_ROCM_ROOT}" "$ENV{HIP_PATH}" "$ENV{ROCM_PATH}")
    if (NOT "${ROCM_DIR}" STREQUAL "")
        list(APPEND CMAKE_PREFIX_PATH "${ROCM_DIR}")
    endif()
endforeach()
message(STATUS "HIP compiler: ${CMAKE_HIP_COMPILER} (${CMAKE_HIP_COMPILER_VERSION}), architectures: ${CMAKE_HIP_ARCHITECTURES}")

include(cmake/libs/hip/hip.cmake)
include(cmake/libs/hip/archs.cmake)
