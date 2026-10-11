include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
set (LIB_NAME FKL)
add_library(${LIB_NAME} INTERFACE)
# add alias so the project can be uses with add_subdirectory
add_library(${LIB_NAME}::${LIB_NAME} ALIAS ${LIB_NAME})

 target_include_directories(
    ${LIB_NAME}
    INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
# FKL's code is C++20. CMake also compiles the CUDA and HIP sources of a target with its C++ standard, unless the target
# sets CUDA_STANDARD or HIP_STANDARD. Requiring cuda_std_20 or hip_std_20 here would break C++ targets of a project that
# adds FKL with add_subdirectory: FKL enables CUDA (HIP) only in its own directory, where its compile features are known
target_compile_features(${LIB_NAME} INTERFACE cxx_std_20)

if (ENABLE_CPU_MULTITHREAD)
    # The backend is part of the build and of the installed package: the thread-pool headers are installed next to
    # FKL's, and consumers of an installed FKL get FK_CPU_MULTITHREAD_ENABLED and Threads (its package configuration
    # has to find_dependency(Threads))
    find_package(Threads REQUIRED)
    target_include_directories(${LIB_NAME} INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/thread-pool/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/fused_kernel/thread_pool>)
    target_link_libraries(${LIB_NAME} INTERFACE Threads::Threads)
    target_compile_definitions(${LIB_NAME} INTERFACE FK_CPU_MULTITHREAD_ENABLED)
    install(FILES ${PROJECT_SOURCE_DIR}/thread-pool/include/ThreadPool.h ${PROJECT_SOURCE_DIR}/thread-pool/include/SafeQueue.h
                  ${PROJECT_SOURCE_DIR}/thread-pool/LICENSE
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fused_kernel/thread_pool)
endif()
# locations are provided by GNUInstallDirs
install(
    TARGETS ${LIB_NAME}
    EXPORT ${LIB_NAME}_Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/fused_kernel DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${PROJECT_SOURCE_DIR}/LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/${LIB_NAME})

# Package configuration, so that installed FKL is used with find_package(FKL) and target FKL::FKL. FKL is header-only,
# so the package does not depend on the architecture and goes to share/
set(FKL_PACKAGE_INSTALL_DIR ${CMAKE_INSTALL_DATADIR}/cmake/${LIB_NAME})
install(EXPORT ${LIB_NAME}_Targets
        FILE ${LIB_NAME}Targets.cmake
        NAMESPACE ${LIB_NAME}::
        DESTINATION ${FKL_PACKAGE_INSTALL_DIR})
configure_package_config_file(${PROJECT_SOURCE_DIR}/cmake/FKLConfig.cmake.in
                              ${CMAKE_CURRENT_BINARY_DIR}/${LIB_NAME}Config.cmake
                              INSTALL_DESTINATION ${FKL_PACKAGE_INSTALL_DIR})
# Before 1.0 a minor version can break the API, so find_package(FKL 0.2) only accepts 0.2.x; from 1.0 on, the same major
# version is compatible
if (PROJECT_VERSION_MAJOR EQUAL 0)
    set(FKL_VERSION_COMPATIBILITY SameMinorVersion)
else()
    set(FKL_VERSION_COMPATIBILITY SameMajorVersion)
endif()
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${LIB_NAME}ConfigVersion.cmake
                                 VERSION ${PROJECT_VERSION}
                                 COMPATIBILITY ${FKL_VERSION_COMPATIBILITY}
                                 ARCH_INDEPENDENT)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${LIB_NAME}Config.cmake ${CMAKE_CURRENT_BINARY_DIR}/${LIB_NAME}ConfigVersion.cmake
        DESTINATION ${FKL_PACKAGE_INSTALL_DIR})

# lib/export is not the installed package: projects that have FKL as a git submodule and never build or install it
# (cvGPUSpeedup, FastNPP) find it with find_package(FKL) in fkl/lib/export, and add fkl/include themselves. Keep it, and
# its version file up to date, until they use add_subdirectory or an installed FKL
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/FKLConfigVersion.cmake.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/export/FKLConfigVersion.cmake"
    @ONLY)
