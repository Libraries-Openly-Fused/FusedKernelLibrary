
check_language(HIP)
list(APPEND CMAKE_PREFIX_PATH "$ENV{HIP_PATH}")

if (NOT CMAKE_HIP_COMPILER)
    message(WARNING "HIP compiler not found. Disabling HIP support.")
    set(ENABLE_HIP OFF CACHE BOOL "Enable HIP support" FORCE)
    return()
endif()

set(CMAKE_CXX_SCAN_FOR_MODULES OFF) #clang with HIP does not come with  clang-scan-deps

# By default enable_language(HIP) sets CMAKE_HIP_ARCHITECTURES to the GPU of this machine (or the compiler default).
# -DHIP_ARCH=<list> (e.g. "gfx1200;gfx1201": Navi 44 and Navi 48 can't load each other's code objects) seeds it instead
# on the first configure of a build directory; an explicit -DCMAKE_HIP_ARCHITECTURES wins over HIP_ARCH.

if(DEFINED HIP_ARCH AND NOT DEFINED CACHE{CMAKE_HIP_ARCHITECTURES})
  set(CMAKE_HIP_ARCHITECTURES "${HIP_ARCH}" CACHE STRING "HIP architectures")
endif()
enable_language(HIP)

include(cmake/libs/hip/hip.cmake)
include(cmake/libs/hip/archs.cmake)
