set (LAUNCH_SOURCES "${CMAKE_SOURCE_DIR}/tests/main.cpp;${CMAKE_SOURCE_DIR}/tests/main.h")
if (WIN32)
    list(APPEND LAUNCH_SOURCES "${CMAKE_SOURCE_DIR}/manifest.xml") #for utf8 codepage
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


function (discover_tests DIR)    
    file(
        GLOB_RECURSE
        TEST_SOURCES
        CONFIGURE_DEPENDS
        "${DIR}/*.h"        
    )
     
    foreach(test_source ${TEST_SOURCES})
         
        get_filename_component(TARGET_NAME ${test_source} NAME_WE)   
        cmake_path(GET test_source  PARENT_PATH  DIR_NAME) #get the directory name of the test source file
        string(FIND ${DIR_NAME} "cudabug"  POS)
        if (${POS} EQUAL -1) #if the directory name does not contain "cudabug"    
            if (${ENABLE_CPU})                    
                add_generated_test("${TARGET_NAME}" "${test_source}" "cpp" "${DIR_NAME}")
            endif()
        endif()
        if (CMAKE_CUDA_COMPILER AND ENABLE_CUDA)
            add_generated_test("${TARGET_NAME}"  "${test_source}" "cu"  "${DIR_NAME}")
            add_cuda_to_test("${TARGET_NAME}_cu")            
        endif()
      
    endforeach()
endfunction()
 