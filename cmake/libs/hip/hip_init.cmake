# HIP (ROCm) backend: FKL's GPU_AMD ParArch, compiled by amdclang++ through CMake's HIP language.
# HIP support is optional: if no HIP compiler is found it is disabled and the other backends are unaffected.
check_language(HIP)

if (NOT CMAKE_HIP_COMPILER)
    message(STATUS "HIP compiler not found, HIP support will be disabled.")
    set(ENABLE_HIP OFF CACHE BOOL "Enable HIP support" FORCE)
    return()
endif()

# By default enable_language(HIP) sets CMAKE_HIP_ARCHITECTURES to the GPU(s) of this machine (or the compiler default).
# -DHIP_ARCH=<list> (e.g. "gfx1200;gfx1201") seeds it instead on the first configure of a build directory; an explicit
# -DCMAKE_HIP_ARCHITECTURES wins over HIP_ARCH.
if (DEFINED HIP_ARCH AND NOT DEFINED CACHE{CMAKE_HIP_ARCHITECTURES})
    set(CMAKE_HIP_ARCHITECTURES "${HIP_ARCH}" CACHE STRING "HIP architectures")
endif()
enable_language(HIP)
# Where find_package(hip) looks for the HIP package: $HIP_PATH / $ROCM_PATH if set, and the ROCm installation of the HIP
# compiler that CMake found
foreach(ROCM_DIR "$ENV{HIP_PATH}" "$ENV{ROCM_PATH}" "${CMAKE_HIP_COMPILER_ROCM_ROOT}")
    if (NOT "${ROCM_DIR}" STREQUAL "")
        list(APPEND CMAKE_PREFIX_PATH "${ROCM_DIR}")
    endif()
endforeach()
message(STATUS "HIP compiler: ${CMAKE_HIP_COMPILER} (${CMAKE_HIP_COMPILER_VERSION}), architectures: ${CMAKE_HIP_ARCHITECTURES}")

include(cmake/libs/hip/hip.cmake)
include(cmake/libs/hip/archs.cmake)
