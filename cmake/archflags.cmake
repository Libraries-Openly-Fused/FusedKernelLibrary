function (add_msvc_flags TARGET_NAME)
    # Add MSVC specific flags for the target
    #CMAKE_SYSTEM_PROCESSOR is broken and windows and return AMD64
    #so we need to use the CMAKE_HOST_PROCESSOR instead
    #https://gitlab.kitware.com/cmake/cmake/-/issues/15170
         
    #message(STATUS "ENV{PROCESSOR_ARCHITECTURE}:" "$ENV{PROCESSOR_ARCHITECTURE}")
    if (${CMAKE_GENERATOR} MATCHES "Ninja")
       set (CMAKE_VS_PLATFORM_NAME "x64" CACHE STRING "Visual Studio platform name" FORCE)
    endif()
    if ( ${CMAKE_VS_PLATFORM_NAME} STREQUAL "x64")
        SET(ARCH_FLAGS "AVX2" CACHE STRING "instruction set to use")
        SET_PROPERTY(CACHE ARCH_FLAGS PROPERTY STRINGS AVX AVX2 AVX512 AVX10.1 disabled)  
        option(ARCH_FLAGS "CPU arch" "AVX2")
            
        if (NOT(${ARCH_FLAGS} STREQUAL "disabled"))                
            target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:${ARCH_FLAGS}>)     
        endif()
    endif()        

    if ( ${CMAKE_VS_PLATFORM_NAME} STREQUAL "ARM64" OR ${CMAKE_VS_PLATFORM_NAME} STREQUAL "ARM64EC")        
        SET(ARCH_FLAGS "armv8.2" CACHE STRING "instruction set to use")
        SET_PROPERTY(CACHE ARCH_FLAGS PROPERTY STRINGS armv8.2 disabled)  
        option(ARCH_FLAGS "CPU arch" "disabled")
        if (NOT(${ARCH_FLAGS} STREQUAL "disabled"))                
            target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:${ARCH_FLAGS}>)                   
           
        endif()
    endif()


endfunction()

#works both for x86_64 and aarch64      
    #we currently don't test avx10.1 with gcc11 on ubuntu 22.04
    #armv9-a (GH100)  not available in gcc11
    #for arm64 we test on jetson orin and grace-hopper
    #sandybridge: minimum avx arch | haswell: minimum avx2 arch | skylake: minium avx512 arch
    #AVX10.1 only gcc 14 (and >15 for diamondrapids)
    
function (add_unix_flags TARGET_NAME)
    if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
        SET(ARCH_FLAGS "native" CACHE STRING "instrucion set to use") 
        SET_PROPERTY(CACHE ARCH_FLAGS PROPERTY STRINGS native sandybridge haswell skylake-avx512 diamondrapids disabled)             
        if (NOT(${ARCH_FLAGS} STREQUAL "disabled"))                
            target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=${ARCH_FLAGS} -mtune=${ARCH_FLAGS}>)  
        endif()
    elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
        SET(ARCH_FLAGS "native" CACHE STRING "instrucion set to use")          
        SET_PROPERTY(CACHE ARCH_FLAGS PROPERTY STRINGS native armv8.2-a armv9-a disabled)   
        option(ARCH_FLAGS "native" "instrucion set to use")          
        if (NOT(${ARCH_FLAGS} STREQUAL "disabled"))                
             target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mcpu=${ARCH_FLAGS}>)  
        endif()                        
        
    endif()
    
endfunction()
# Add architecture-specific optimization flags for the target
function(add_optimization_flags TARGET_NAME)    
    if (MSVC)        
        add_msvc_flags(${TARGET_NAME})
    elseif (UNIX)             
        add_unix_flags(${TARGET_NAME})    
    endif()
endfunction()