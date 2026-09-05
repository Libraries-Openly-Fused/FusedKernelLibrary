
check_language(HIP)

if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
            message(STATUS "HIP_PATH: " ${HIP_PATH})
        else()
            set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
            message(STATUS "HIP_PATH: " ${HIP_PATH})
    endif()
endif()
    #set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})

if (NOT DEFINED ROCM_PATH )
    set (ROCM_PATH ${HIP_PATH}  CACHE STRING "Default ROCM installation directory." )
else()
    set (ROCM_PATH $ENV{HIP_PATH}  CACHE STRING "Default ROCM installation directory." )
endif()

set(CMAKE_HIP_COMPILER_ROCM_ROOT ${ROCM_PATH})
# Search for rocm in common locations
list(APPEND CMAKE_PREFIX_PATH ${HIP_PATH}/cmake ${HIP_PATH}/lib/cmake/hip-lang ${ROCM_PATH})
message(STATUS "CMAKE_PREFIX_PATH: " ${CMAKE_PREFIX_PATH})

find_package(hip-lang CONFIG REQUIRED)
message(STATUS "Found hip-lang")

if (CMAKE_HIP_COMPILER)       
    message("Using HIP compiler at " ${CMAKE_HIP_COMPILER})    
    set(CMAKE_CXX_SCAN_FOR_MODULES OFF) #clang with HIP does not come with  clang-scan-deps       
    enable_language(HIP)
else()
    if (NOT (ENABLE_CPU))
        message(FATAL_ERROR "HIP compiler not found. the build cannot proceed. ")        
    else ()
        message(WARNING "HIP compiler not found. Disabling HIP support.")
        set(ENABLE_HIP OFF CACHE BOOL "Enable HIP support" FORCE)
        return()
    endif()            
endif()

include(cmake/libs/hip/hip.cmake)
include(cmake/libs/hip/archs.cmake)
