set (LAUNCH_SOURCES "${CMAKE_SOURCE_DIR}/tests/main.cpp;${CMAKE_SOURCE_DIR}/tests/main.h")
if (WIN32)
    list(APPEND LAUNCH_SOURCES "${CMAKE_SOURCE_DIR}/utf8cp.manifest") #for utf8 codepage
endif() 

function(add_cuda_to_test TARGET_NAME)
    add_cuda_to_target(${TARGET_NAME} "")
    set_target_cuda_arch_flags(${TARGET_NAME})
        
    if(${ENABLE_DEBUG})
        add_cuda_debug_support_to_target(${TARGET_NAME})
    endif()
    if(${ENABLE_NVTX})
        add_nvtx_support_to_target(${TARGET_NAME})
    endif()
endfunction()

function(configure_test_target_flags TARGET_NAME TEST_SOURCE DIR)
        
        set(TEST_GENERATED_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_${EXTENSION}/launcher.${EXTENSION}") #use the same name as the target	)			       
        configure_file(${CMAKE_SOURCE_DIR}/tests/launcher.in ${TEST_GENERATED_SOURCE} @ONLY) #replace variables in the test source file                         
        #message(STATUS "Adding test: ${TARGET_NAME} from ${TEST_GENERATED_SOURCE} and ${TEST_SOURCE}")
         
        if(${ENABLE_BENCHMARK})
            target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_BENCHMARK)
        endif()
        set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)            
        target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_SOURCE_DIR}")        
        target_include_directories(${TARGET_NAME} PUBLIC "${DIR}")      
        target_link_libraries(${TARGET_NAME} PUBLIC FKL::FKL)
        if (MSVC)
            target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/diagnostics:caret>)
             target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
        #    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
             #target_link_libraries(${TARGET_NAME} -manifest:embed -manifestinput:"${PROJECT_SOURCE_DIR}/myapp.manifest" 
        endif()
        if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wno-c++11-narrowing>)
        
        #    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wconversion>)
            
        endif()

        add_optimization_flags(${TARGET_NAME})
        
       
endfunction()


function (add_generated_lib TARGET_NAME TEST_SOURCE EXTENSION DIR)                       
        set (TARGET_NAME "${TARGET_NAME}")
        add_library(${TARGET_NAME} SHARED "${TEST_GENERATED_SOURCE};${TEST_SOURCE}" )
        set_target_properties(${TARGET_NAME}  PROPERTIES LINKER_LANGUAGE CXX)
      #  target_sources(${TARGET_NAME} PRIVATE ${LAUNCH_SOURCES})      
        configure_test_target_flags("${TARGET_NAME}" "${TEST_SOURCE}" "${DIR}")  
        set_property(TARGET "${TARGET_NAME}" PROPERTY FOLDER "${DIR}/")  

endfunction()

function (add_generated_test TARGET_NAME TEST_SOURCE EXTENSION DIR)                       
        set(TEST_GENERATED_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_${EXTENSION}/launcher.${EXTENSION}") #use the same name as the target	)			
        set (TARGET_NAME_EXT "${TARGET_NAME}_${EXTENSION}")
        configure_file(${CMAKE_SOURCE_DIR}/tests/launcher.in ${TEST_GENERATED_SOURCE} @ONLY) #replace variables in the test source file                 
        set (TARGET_NAME_EXT "${TARGET_NAME}_${EXTENSION}")
        
        #message(STATUS "Adding test: ${TARGET_NAME_EXT} from ${TEST_GENERATED_SOURCE} and ${TEST_SOURCE}")
        add_executable(${TARGET_NAME_EXT} "${TEST_GENERATED_SOURCE};${TEST_SOURCE}" )
        configure_test_target_flags("${TARGET_NAME_EXT}" "${TEST_SOURCE}"  "${DIR}")
        target_sources(${TARGET_NAME_EXT} PRIVATE ${LAUNCH_SOURCES})      
        add_test(NAME  ${TARGET_NAME_EXT} COMMAND ${TARGET_NAME_EXT})          
        cmake_path(SET path2 "${DIR}")
		cmake_path(GET path2 FILENAME DIR_NAME)   
        cmake_path(GET path2 PARENT_PATH DIR_PARENT_PATH)       
        if (${EXTENSION} STREQUAL "cu")
            set(FKL_BACKEND "cuda")
        elseif(${EXTENSION} STREQUAL "cpp")  
            set(FKL_BACKEND "cpu")
        else()
            message(FATAL_ERROR "Unknown extension: ${EXTENSION}") 
        endif()

        set_property(TARGET "${TARGET_NAME_EXT}" PROPERTY FOLDER "${DIR_PARENT_PATH}/${FKL_BACKEND}/")    
        
endfunction()
