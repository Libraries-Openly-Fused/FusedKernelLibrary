# How the device code of an executable is split in modules (-fsycl-device-code-split). The device compiles a module,
# at run time, the first time one of its kernels is used. With per_kernel every kernel is a module: only the kernels
# that are used are compiled, and a kernel the device can not compile does not affect the others. FKL instantiates
# kernels it may never launch (for instance, a TransformDPP with thread fusion also has the kernel without it), so
# with the other values (per_source, off, auto) the device also compiles those.
set(SYCL_DEVICE_CODE_SPLIT "per_kernel" CACHE STRING "SYCL device code split of the SYCL targets")
set_property(CACHE SYCL_DEVICE_CODE_SPLIT PROPERTY STRINGS per_kernel per_source off auto)

# Options for the compiler of the device, the one that compiles each kernel at run time (-Xsycl-target-backend). By
# default, single precision divisions and square roots are correctly rounded, as they are on the CPU and with CUDA:
# without this option the compiler of the Intel GPUs uses faster ones that are not (the SPIR-V environment of OpenCL
# allows an error of 2.5 ulp), and the results are not the ones of the other backends.
set(SYCL_DEVICE_COMPILER_OPTIONS "-cl-fp32-correctly-rounded-divide-sqrt" CACHE STRING
    "Options for the device compiler of the SYCL targets")

# Compiles the C++ sources of the target as SYCL, and links it as a SYCL program. The name does not collide with
# add_sycl_to_target of the IntelSYCL package of oneAPI.
function(fkl_add_sycl_to_target TARGET_NAME)
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fsycl>)
    target_link_options(${TARGET_NAME} PRIVATE -fsycl -fsycl-device-code-split=${SYCL_DEVICE_CODE_SPLIT})
    if (SYCL_DEVICE_COMPILER_OPTIONS)
        target_link_options(${TARGET_NAME} PRIVATE
                            "SHELL:-Xsycl-target-backend \"${SYCL_DEVICE_COMPILER_OPTIONS}\"")
    endif()
endfunction()
