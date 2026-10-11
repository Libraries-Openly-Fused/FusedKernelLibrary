set (LIB_NAME FKL)
add_library(${LIB_NAME} INTERFACE)
# add alias so the project can be uses with add_subdirectory
add_library(${LIB_NAME}::${LIB_NAME} ALIAS ${LIB_NAME})

 target_include_directories(
    ${LIB_NAME}
    INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

if (ENABLE_CPU_MULTITHREAD)
    find_package(Threads REQUIRED)
    target_include_directories(${LIB_NAME} INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/thread-pool/include>)
    target_link_libraries(${LIB_NAME} INTERFACE $<BUILD_INTERFACE:Threads::Threads>)
    target_compile_definitions(${LIB_NAME} INTERFACE $<BUILD_INTERFACE:FK_CPU_MULTITHREAD_ENABLED>)
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