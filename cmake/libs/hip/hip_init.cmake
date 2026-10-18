
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
set(HIP_ARCH "gfx1200;gfx1201" CACHE STRING "HIP architecture(s) to build")
if (NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "${HIP_ARCH}" CACHE STRING "HIP architectures")
    set(FK_HIP_ARCH_SEEDED "${HIP_ARCH}" CACHE INTERNAL "HIP_ARCH value CMAKE_HIP_ARCHITECTURES was seeded from")
elseif (DEFINED FK_HIP_ARCH_SEEDED AND NOT "${HIP_ARCH}" STREQUAL "${FK_HIP_ARCH_SEEDED}"
        AND "${CMAKE_HIP_ARCHITECTURES}" STREQUAL "${FK_HIP_ARCH_SEEDED}")
    set(CMAKE_HIP_ARCHITECTURES "${HIP_ARCH}" CACHE STRING "HIP architectures" FORCE)
    set(FK_HIP_ARCH_SEEDED "${HIP_ARCH}" CACHE INTERNAL "HIP_ARCH value CMAKE_HIP_ARCHITECTURES was seeded from")
endif()
enable_language(HIP)

include(cmake/libs/hip/hip.cmake)
include(cmake/libs/hip/archs.cmake)
