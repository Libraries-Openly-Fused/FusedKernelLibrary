include(GNUInstallDirs)
set (LIB_NAME FKL)
add_library(${LIB_NAME} INTERFACE)
# add alias so the project can be uses with add_subdirectory
add_library(${LIB_NAME}::${LIB_NAME} ALIAS ${LIB_NAME})

 target_include_directories(
    ${LIB_NAME}
    INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

if (ENABLE_CPU_MULTITHREAD)
    # The backend is part of the build and of the installed package: the thread-pool headers are installed next to
    # FKL's, and consumers of an installed FKL get FK_CPU_MULTITHREAD_ENABLED and Threads (its package configuration
    # has to find_dependency(Threads))
    find_package(Threads REQUIRED)
    target_include_directories(${LIB_NAME} INTERFACE
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/thread-pool/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/fused_kernel/thread_pool>)
    target_link_libraries(${LIB_NAME} INTERFACE Threads::Threads)
    target_compile_definitions(${LIB_NAME} INTERFACE FK_CPU_MULTITHREAD_ENABLED)
    install(FILES ${CMAKE_SOURCE_DIR}/thread-pool/include/ThreadPool.h ${CMAKE_SOURCE_DIR}/thread-pool/include/SafeQueue.h
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fused_kernel/thread_pool)
endif()
# locations are provided by GNUInstallDirs
install(
    TARGETS ${LIB_NAME}
    EXPORT ${LIB_NAME}_Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

#always update exported version file when the version changes
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/FKLConfigVersion.cmake.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/export/FKLConfigVersion.cmake"
    @ONLY)