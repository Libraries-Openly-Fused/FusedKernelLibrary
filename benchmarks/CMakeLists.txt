set (LAUNCH_SOURCES "${CMAKE_SOURCE_DIR}/tests/main.cpp;${CMAKE_SOURCE_DIR}/tests/main.h")
if (WIN32)
    list(APPEND LAUNCH_SOURCES "${CMAKE_SOURCE_DIR}/utf8cp.manifest") #for utf8 codepage
endif() 

function(add_cuda_to_benchmark TARGET_NAME)
    add_cuda_to_target(${TARGET_NAME} "")
    set_target_cuda_arch_flags(${TARGET_NAME})
        
    if(${ENABLE_DEBUG})
        add_cuda_debug_support_to_target(${TARGET_NAME})
    endif()
    if(${ENABLE_NVTX})
        add_nvtx_support_to_target(${TARGET_NAME})
    endif()
endfunction()

function (add_generated_benchmark TARGET_NAME TEST_SOURCE EXTENSION DIR)
                       
        set(TEST_GENERATED_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_${EXTENSION}/launcher.${EXTENSION}") #use the same name as the target	)			
       
        configure_file(${CMAKE_SOURCE_DIR}/benchmarks/launcher.in ${TEST_GENERATED_SOURCE} @ONLY) #replace variables in the test source file                 
        set (TARGET_NAME_EXT "${TARGET_NAME}_${EXTENSION}")
        #message(STATUS "Adding test: ${TARGET_NAME_EXT} from ${TEST_GENERATED_SOURCE} and ${TEST_SOURCE}")
        add_executable(${TARGET_NAME_EXT} "${TEST_GENERATED_SOURCE};${TEST_SOURCE};${LAUNCH_SOURCES}" )
        
        if(${ENABLE_BENCHMARK})
            target_compile_definitions(${TARGET_NAME_EXT} PRIVATE ENABLE_BENCHMARK)
        endif()
        
         
        #todo: add hip support
        set_target_properties(${TARGET_NAME_EXT} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)            
        target_include_directories(${TARGET_NAME_EXT} PRIVATE "${CMAKE_SOURCE_DIR}")        
        target_include_directories(${TARGET_NAME_EXT} PRIVATE "${DIR}")      
        target_link_libraries(${TARGET_NAME_EXT} PRIVATE FKL::FKL)
        if (MSVC)
            target_compile_options(${TARGET_NAME_EXT} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/diagnostics:caret>)
            target_compile_options(${TARGET_NAME_EXT} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>)
        endif()
        
        add_optimization_flags(${TARGET_NAME_EXT})
        
        add_test(NAME  ${TARGET_NAME_EXT} COMMAND ${TARGET_NAME_EXT})    
        cmake_path(SET path2 "${DIR}")
		cmake_path(GET path2 FILENAME DIR_NAME)       
		set_property(TARGET ${cuda_target} PROPERTY FOLDER benchmarks/${DIR_NAME})
        set_property(TARGET "${TARGET_NAME_EXT}" PROPERTY FOLDER "benchmarks/${EXTENSION}/${DIR_NAME}")    
        
endfunction()

function (discover_benchmark DIR)
    file(
        GLOB_RECURSE
        BENCHMARK_SOURCES
        CONFIGURE_DEPENDS
        "${DIR}/*.h"
    )
    
    foreach(benchmark_source ${BENCHMARK_SOURCES})
        get_filename_component(TARGET_NAME ${benchmark_source} NAME_WE)    
		cmake_path(GET benchmark_source  PARENT_PATH  DIR_NAME) #get the directory name of the test source file
        if (${ENABLE_CPU})
            add_generated_benchmark("${TARGET_NAME}" "${benchmark_source}" "cpp" "${DIR_NAME}")
        endif()
		if (CMAKE_CUDA_COMPILER AND ENABLE_CUDA)
            add_generated_benchmark("${TARGET_NAME}"  "${benchmark_source}" "cu"  "${DIR_NAME}")
            add_cuda_to_benchmark("${TARGET_NAME}_cu")            
        endif()
    endforeach()
endfunction()

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUT_DIR})

MACRO(SUBDIRLIST result curdir)
FILE(GLOB children ${curdir}/*) # 
SET(dirlist "")
FOREACH(child ${children})
    IF(IS_DIRECTORY ${child}) 
        LIST(APPEND dirlist ${child})
    ENDIF()
ENDFOREACH()
SET(${result} ${dirlist} )
ENDMACRO()

set (LIST_OF_DIRS "")
SUBDIRLIST(LIST_DIRS ${CMAKE_CURRENT_SOURCE_DIR})
 
foreach(DIR ${LIST_DIRS})
    discover_benchmark(${DIR})    
endforeach()