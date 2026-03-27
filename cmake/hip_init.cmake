enable_language(HIP)

# Set HIP compiler and standard
set(CMAKE_HIP_COMPILER "hipcc")
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

include(cmake/libs/hip/hip.cmake)
include(cmake/libs/hip/archs.cmake)
