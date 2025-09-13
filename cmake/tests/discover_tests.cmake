include (${CMAKE_SOURCE_DIR}/cmake/tests/add_generated_test.cmake)
 

function (discover_tests DIR)    
    file(
        GLOB_RECURSE
        TEST_SOURCES
        CONFIGURE_DEPENDS
        "${DIR}/*.h"        
    )
    list(FILTER TEST_SOURCES EXCLUDE REGEX ".*_common.*") 
    foreach(TEST_SOURCE ${TEST_SOURCES})         
        get_filename_component(TARGET_NAME ${TEST_SOURCE} NAME_WE)           
        file (READ ${TEST_SOURCE} TEST_SOURCE_CONTENTS ) #read the contents of the test source file
       
        string(FIND "${TEST_SOURCE_CONTENTS}" "ONLY_CU"  POS_ONLY_CU)
        string(FIND "${TEST_SOURCE_CONTENTS}" "ONLY_CPU"  POS_ONLY_CPU)
        set (TEST_SOURCE1 "${TEST_SOURCE}")
        string(REPLACE ${CMAKE_SOURCE_DIR} "" TEST_SOURCE1 "${TEST_SOURCE1}") #make the path relative to the current directory)
        
        cmake_path(GET TEST_SOURCE1 RELATIVE_PART DIR_RELATIVE_PATH) 
              
        if (${POS_ONLY_CU} EQUAL -1) #if the source file does not contain "__ONLY_CU__"    
            if (${ENABLE_CPU})                                    
                add_generated_test("${TARGET_NAME}" "${TEST_SOURCE}" "cpp" "${DIR_RELATIVE_PATH}")                
             endif()
        endif()

        if (CMAKE_CUDA_COMPILER AND ENABLE_CUDA)
            if (${POS_ONLY_CPU} EQUAL -1) #if the source file does not contain "__ONLY_CPU__"         
                add_generated_test("${TARGET_NAME}"  "${TEST_SOURCE}" "cu"  "${DIR_RELATIVE_PATH}")
                add_cuda_to_test("${TARGET_NAME}_cu")                           
            endif()
        endif()         
    endforeach()   
endfunction()
 