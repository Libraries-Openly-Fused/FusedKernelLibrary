# The SYCL backend (ParArch::GPU_INTEL). CMake has no SYCL language: the C++ compiler has to be a SYCL compiler
# (DPC++: icx, icpx, or clang++ with SYCL support), and the targets that use the backend are compiled and linked with
# -fsycl (see cmake/libs/sycl/sycl.cmake). The other C++ targets are not, and use the CPU backends.
include(CheckCXXSourceCompiles)
include(CMakePushCheckState)

cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_FLAGS "-fsycl")
set(CMAKE_REQUIRED_LINK_OPTIONS "-fsycl")
set(CMAKE_REQUIRED_QUIET ON)
check_cxx_source_compiles("
#include <sycl/sycl.hpp>
#if !defined(SYCL_LANGUAGE_VERSION)
#error SYCL_LANGUAGE_VERSION is not defined
#endif
int main() {
    return static_cast<int>(sizeof(sycl::queue)) == 0;
}
" FKL_CXX_COMPILER_COMPILES_SYCL)
cmake_pop_check_state()

if (NOT FKL_CXX_COMPILER_COMPILES_SYCL)
    # Not a warning: unlike nvcc or the HIP compiler, which are found or not next to any C++ compiler, SYCL is a
    # property of the C++ compiler that was chosen
    message(STATUS "SYCL backend (ENABLE_SYCL): OFF (the C++ compiler does not compile SYCL with -fsycl)")
    set(ENABLE_SYCL OFF CACHE BOOL "Enable SYCL support" FORCE)
    return()
endif()

message(STATUS "SYCL backend (ENABLE_SYCL): ON (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")

include(cmake/libs/sycl/sycl.cmake)
