
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

#cuda is optional, but if it is found, it will be used
option(ENABLE_CPU "Enable CPU support" ON)
option(ENABLE_CUDA "Enable CUDA support" ON)
option(ENABLE_HIP "Enable HIP support" ON)


include(CheckLanguage)

if (ENABLE_CPU)
    check_language(CXX)
    if (CMAKE_CXX_COMPILER)
         enable_language(CXX)
    #    include(cmake/cpu_init.cmake)
    else()
    #    message(FATAL_ERROR "C++ compiler not found. the build cannot proceed.")
    endif()
endif()

if (ENABLE_CUDA)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        include(cmake/cuda_init.cmake)
    else()
    if (NOT (ENABLE_CPU))
            message(FATAL_ERROR "CUDA compiler not found. the build cannot proceed.")
        else ()
            message(WARNING "CUDA compiler not found. Disabling CUDA support.")
            set(ENABLE_CUDA OFF CACHE BOOL "Enable CUDA support" FORCE)
        endif()        
        
    endif()
endif()

if (ENABLE_HIP)    
    include(cmake/hip_init.cmake)
    
endif()

if (ENABLE_CUDA AND ENABLE_HIP)
    message(FATAL_ERROR "ENABLE_CUDA and ENABLE_HIP cannot be enabled at the same time.")
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
