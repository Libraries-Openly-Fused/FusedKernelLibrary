include (${CMAKE_SOURCE_DIR}/cmake/tests/add_generated_test.cmake)

function (discover_tests DIR)    
    file(
        GLOB_RECURSE
        TEST_SOURCES
        CONFIGURE_DEPENDS
        "${DIR}/*.h"        
    )
     
    foreach(test_source ${TEST_SOURCES})         
        get_filename_component(TARGET_NAME ${test_source} NAME_WE)           
        file (READ ${test_source} TEST_SOURCE_CONTENTS ) #read the contents of the test source file
       
        string(FIND "${TEST_SOURCE_CONTENTS}" "ONLY_CU"  POS_ONLY_CU)
        string(FIND "${TEST_SOURCE_CONTENTS}" "ONLY_CPU"  POS_ONLY_CPU)
        string(FIND "${TEST_SOURCE_CONTENTS}" "LLVM_JIT"  POS_LLVM_JIT)
        cmake_path(GET test_source RELATIVE_PART DIR_RELATIVE_PATH)     
        string(REPLACE "${PROJECT_NAME}/" " " DIR_RELATIVE_PATH "${DIR_RELATIVE_PATH}") #remove the project name from the relative path
        
        if (${POS_ONLY_CU} EQUAL -1) #if the source file does not contain "__ONLY_CU__"    
            if (${ENABLE_CPU})                    
                add_generated_test("${TARGET_NAME}" "${test_source}" "cpp" "${DIR_RELATIVE_PATH}" ${POS_LLVM_JIT})                
            endif()
        endif()

        if (CMAKE_CUDA_COMPILER AND ENABLE_CUDA)
            if (${POS_ONLY_CPU} EQUAL -1) #if the source file does not contain "__ONLY_CPU__"
                add_generated_test("${TARGET_NAME}"  "${test_source}" "cu"  "${DIR_RELATIVE_PATH}" ${POS_LLVM_JIT})
                add_cuda_to_test("${TARGET_NAME}_cu")                           
            endif()
        endif()
         
      
    endforeach()
endfunction()
 