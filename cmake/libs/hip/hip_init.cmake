
check_language(HIP)
list(APPEND CMAKE_PREFIX_PATH "$ENV{HIP_PATH}")

if (NOT CMAKE_HIP_COMPILER)
    message(WARNING "HIP compiler not found. Disabling HIP support.")
    set(ENABLE_HIP OFF CACHE BOOL "Enable HIP support" FORCE)
    return()
endif()

set(CMAKE_CXX_SCAN_FOR_MODULES OFF) #clang with HIP does not come with  clang-scan-deps

# HIP_ARCH seeds CMAKE_HIP_ARCHITECTURES before enable_language(HIP), which would otherwise set it to the GPU of this
# machine (or the compiler default) and leave HIP_ARCH unused. gfx1200 (Navi 44) and gfx1201 (Navi 48) can't load each
# other's code objects, so both are built. Passing CMAKE_HIP_ARCHITECTURES explicitly still wins; changing HIP_ARCH on
# a reconfigure updates it unless CMAKE_HIP_ARCHITECTURES was changed by hand.

if(DEFINED HIP_ARCH AND NOT DEFINED CACHE{CMAKE_HIP_ARCHITECTURES})
  set(CMAKE_HIP_ARCHITECTURES "${HIP_ARCH}" CACHE STRING "HIP architectures")
endif()
enable_language(HIP)

include(cmake/libs/hip/hip.cmake)
include(cmake/libs/hip/archs.cmake)
