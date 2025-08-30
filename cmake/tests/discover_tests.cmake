include (${CMAKE_SOURCE_DIR}/cmake/tests/add_generated_test.cmake)
 
function (add_shared_target TARGET_BASE_NAME EXTENSION FUNDAMENTAL_TYPE SOURCES)
      set(TARGET_NAME "${TARGET_BASE_NAME}_${FUNDAMENTAL_TYPE}")         
        add_generated_lib("${TARGET_NAME}_${EXTENSION}" "${SOURCES}"  "/utests/shared/${EXTENSION}")                         
        target_include_directories("${TARGET_NAME}_${EXTENSION}" PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include") 
        target_include_directories("${TARGET_NAME}_${EXTENSION}" PUBLIC "${CMAKE_SOURCE_DIR}/utests")         
        target_link_libraries(${TARGET_BASE_NAME}_${EXTENSION} PRIVATE "${TARGET_NAME}_${EXTENSION}")
endfunction()

function (add_shared_test_libs TARGET_BASE_NAME)
    set (FUNDAMENTAL_TYPES uchar char ushort short uint int ulong long ulonglong longlong float double) 
    foreach(FUNDAMENTAL_TYPE ${FUNDAMENTAL_TYPES})   
        set(TARGET_NAME "${TARGET_BASE_NAME}_${FUNDAMENTAL_TYPE}")    
        set(SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/include/${TARGET_NAME}.h"        
        "${CMAKE_SOURCE_DIR}/utests/utest_common.h"        
        
        )
        set(TARGET_NAME "${TARGET_BASE_NAME}_${FUNDAMENTAL_TYPE}")   

        if (${ENABLE_CPU})               
            set(EXTENSION cpp)     
                   
            add_shared_target("${TARGET_BASE_NAME}" "${EXTENSION}" "${FUNDAMENTAL_TYPE}" 
            "${SOURCES};${CMAKE_CURRENT_SOURCE_DIR}/${EXTENSION}/${TARGET_NAME}.${EXTENSION};${CMAKE_BINARY_DIR}/exports/${TARGET_NAME}_export.h;")
            add_generated_export_header_to_target("${TARGET_NAME}_${EXTENSION}" "${TARGET_BASE_NAME}_${FUNDAMENTAL_TYPE}")
        endif()
        if (CMAKE_CUDA_COMPILER AND ENABLE_CUDA)    
            set(EXTENSION_CU cu)      
            #some targets don't have cuda equivalent                      
            if (TARGET ${TARGET_BASE_NAME}_${EXTENSION_CU})                                               
                add_shared_target("${TARGET_BASE_NAME}" "${EXTENSION_CU}" "${FUNDAMENTAL_TYPE}" 
                "${SOURCES};${CMAKE_CURRENT_SOURCE_DIR}/${EXTENSION_CU}/${TARGET_NAME}.${EXTENSION_CU};${CMAKE_BINARY_DIR}/exports//${TARGET_NAME}_export.h;")
                add_cuda_to_test("${TARGET_NAME}_${EXTENSION_CU}")   
                add_generated_export_header_to_target("${TARGET_NAME}_${EXTENSION_CU}" "${TARGET_BASE_NAME}_${FUNDAMENTAL_TYPE}")
            endif()                                
        endif()
    endforeach()  
endfunction()

function (discover_tests DIR)    
    file(
        GLOB_RECURSE
        TEST_SOURCES
        CONFIGURE_DEPENDS
        "${DIR}/*.h"        
    )
     list(FILTER TEST_SOURCES EXCLUDE REGEX ".*_shared.*")  
    foreach(TEST_SOURCE ${TEST_SOURCES})         
        get_filename_component(TARGET_NAME ${TEST_SOURCE} NAME_WE)           
        file (READ ${TEST_SOURCE} TEST_SOURCE_CONTENTS ) #read the contents of the test source file
       
        string(FIND "${TEST_SOURCE_CONTENTS}" "ONLY_CU"  POS_ONLY_CU)
        string(FIND "${TEST_SOURCE_CONTENTS}" "ONLY_CPU"  POS_ONLY_CPU)
        cmake_path(GET TEST_SOURCE RELATIVE_PART DIR_RELATIVE_PATH)     
        string(REPLACE "${PROJECT_NAME}/" " " DIR_RELATIVE_PATH "${DIR_RELATIVE_PATH}") #remove the project name from the relative path
        
        if (${POS_ONLY_CU} EQUAL -1) #if the source file does not contain "__ONLY_CU__"    
            if (${ENABLE_CPU})                       
                add_generated_test("${TARGET_NAME}" "${TEST_SOURCE}" "cpp" "${DIR_RELATIVE_PATH}")                
             endif()
        endif()

        if (CMAKE_CUDA_COMPILER AND ENABLE_CUDA)
            if (${POS_ONLY_CPU} EQUAL -1) #if the source file does not contain "__ONLY_CPU__"
          #  message(STATUS   "Adding test: ${TARGET_NAME}_cu from ${TEST_SOURCE}")
                add_generated_test("${TARGET_NAME}"  "${TEST_SOURCE}" "cu"  "${DIR_RELATIVE_PATH}")
                add_cuda_to_test("${TARGET_NAME}_cu")                           
            endif()
        endif()         
    endforeach()   
endfunction()
 