
check_language(HIP)
list(APPEND CMAKE_PREFIX_PATH "$ENV{HIP_PATH}/hip" "$ENV{HIP_PATH}")
 
if (NOT CMAKE_HIP_COMPILER)        
    message(WARNING "HIP compiler not found. Disabling HIP support.")
    set(ENABLE_HIP OFF CACHE BOOL "Enable HIP support" FORCE)
    return()    
endif()

set(CMAKE_CXX_SCAN_FOR_MODULES OFF) #clang with HIP does not come with  clang-scan-deps       
enable_language(HIP)
find_package(HIP CONFIG REQUIRED)     
  
include(cmake/libs/hip/hip.cmake)
include(cmake/libs/hip/archs.cmake)
