
cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

set (PROJECT_VERSION_MAJOR 0)
set (PROJECT_VERSION_MINOR 1)
set (PROJECT_VERSION_REV 14)
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

#cuda is optional, but if it is found, it will be used
option(ENABLE_CPU "Enable CPU support" ON)
include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    option (ENABLE_CUDA "Enable CUDA support" ON)
    if (${ENABLE_CUDA})
        include(cmake/cuda_init.cmake)
    endif()
else()
    message(STATUS "CUDA compiler not found, CUDA support will be disabled.")     
endif()

#hip is optional too: if amdclang++ is found, the GPU_AMD backend is built. Only the amd HIP platform is considered:
#HIP on NVIDIA GPUs is not a FKL backend (CUDA is).
if (NOT DEFINED CMAKE_HIP_PLATFORM)
    set(CMAKE_HIP_PLATFORM "amd")
endif()
check_language(HIP)
if (CMAKE_HIP_COMPILER AND "${CMAKE_HIP_PLATFORM}" STREQUAL "amd")
    option (ENABLE_HIP "Enable HIP (ROCm) support" ON)
    if (${ENABLE_HIP})
        include(cmake/libs/hip/hip_init.cmake)
    endif()
else()
    message(STATUS "HIP compiler for AMD GPUs not found, HIP support will be disabled.")
    # Not a cache variable: an ENABLE_HIP=ON given by the user is kept for a configure that finds the compiler
    set(ENABLE_HIP OFF)
endif()

# Multithreaded CPU backend (ParArch::CPU_MULTITHREAD), which uses the thread pool of the thread-pool submodule
if (EXISTS ${PROJECT_SOURCE_DIR}/thread-pool/include/ThreadPool.h)
    option(ENABLE_CPU_MULTITHREAD "Enable the multithreaded CPU backend (ParArch::CPU_MULTITHREAD)" ON)
else()
    message(STATUS "thread-pool submodule not found (git submodule update --init): CPU_MULTITHREAD backend disabled")
    option(ENABLE_CPU_MULTITHREAD "Enable the multithreaded CPU backend (ParArch::CPU_MULTITHREAD)" OFF)
endif()
if (ENABLE_CPU_MULTITHREAD AND NOT EXISTS ${PROJECT_SOURCE_DIR}/thread-pool/include/ThreadPool.h)
    message(FATAL_ERROR "ENABLE_CPU_MULTITHREAD needs the thread-pool submodule: git submodule update --init")
endif()
message(STATUS "CPU_MULTITHREAD backend (ENABLE_CPU_MULTITHREAD): ${ENABLE_CPU_MULTITHREAD}")
if (NOT ENABLE_CPU_MULTITHREAD AND EXISTS ${PROJECT_SOURCE_DIR}/thread-pool/include/ThreadPool.h)
    message(STATUS "The thread-pool submodule is checked out: configure with -DENABLE_CPU_MULTITHREAD=ON to enable it")
endif()

add_subdirectory(include)
add_subdirectory(lib)

option (BUILD_TEST "build standard  tests" ON)
option (BUILD_UTEST "build standard unit tests" ON)

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
