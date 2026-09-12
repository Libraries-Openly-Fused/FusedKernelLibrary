
cmake_minimum_required(VERSION 3.28 FATAL_ERROR)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)
set (PROJECT_VERSION_MAJOR 0)
set (PROJECT_VERSION_MINOR 2)
set (PROJECT_VERSION_REV 1)
set (PROJECT_VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_REV})
include (cmake/cmake_init.cmake)
include (cmake/doxygen.cmake)
include (cmake/deploy/deploy_dependencies.cmake)

include (cmake/archflags.cmake)
include (cmake/generators/version_header.cmake)
include (cmake/generators/export_header.cmake)
project(FusedKernelLibrary VERSION ${PROJECT_VERSION} LANGUAGES CXX  
DESCRIPTION "Implementation of a methodology that allows all sorts of user defined GPU kernel fusion, for non CUDA programmers." 
HOMEPAGE_URL "https://github.com/morousg/FusedKernelLibrary" )

# Options for enabling different types of GPU support
option(ENABLE_CPU "Enable CPU support" ON)
option(ENABLE_CUDA "Enable CUDA support" ON)
option(ENABLE_HIP "Enable HIP support" ON)
include(CheckLanguage)
if (ENABLE_CUDA)
    include(cmake/libs/cuda/cuda_init.cmake)
    endif()

if (ENABLE_HIP)
    include(cmake/libs/hip/hip_init.cmake)
endif()

add_subdirectory(include)
add_subdirectory(lib)

option (BUILD_TEST "build standard  tests" ON)
option (BUILD_UTEST "build standard unit tests" ON)
option (ENABLE_BENCHMARK "build benchmarking unit tests" OFF)

if (${BUILD_TEST})    
    include (cmake/tests/discover_tests.cmake)
    enable_testing()  
    add_subdirectory(tests)    
endif()

if (${BUILD_UTEST})    
    include (cmake/tests/discover_tests.cmake)
    enable_testing()      
    add_subdirectory(utests)         
endif()

if (${ENABLE_BENCHMARK})
    include (cmake/tests/discover_tests.cmake)
    enable_testing()
    add_subdirectory(benchmarks)
endif()
