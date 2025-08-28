include (${CMAKE_SOURCE_DIR}/cmake/tests/add_generated_test.cmake)
function (discover_gen_libs DIR LIBS_FUNDAMENTAL_TYPE)
    file(
            GLOB_RECURSE
            TEST_SOURCES
            CONFIGURE_DEPENDS
            "${DIR}/*.h.in"        
        )
        list(GET  TEST_SOURCES 0 TEST_SOURCE_H)
       
       file(
            GLOB_RECURSE
            TEST_SOURCES
            CONFIGURE_DEPENDS
            "${DIR}/*.cpp.in"        
        )

        list(GET  TEST_SOURCES 0 TEST_SOURCE_CPP)
        
        get_filename_component(TARGET_NAME ${TEST_SOURCE_CPP} NAME_WE)                   
        cmake_path(GET TEST_SOURCE_CPP RELATIVE_PART DIR_RELATIVE_PATH)     
        string(REPLACE "${PROJECT_NAME}/" " " DIR_RELATIVE_PATH "${DIR_RELATIVE_PATH}") #remove the project name from the relative path
        set(LIBS_FUNDAMENTAL_TYPE1 "" )
      
        set (FUNDAMENTAL_TYPES uchar char ushort short uint int ulong long ulonglong longlong float double) 
        foreach(FUNDAMENTAL_TYPE ${FUNDAMENTAL_TYPES})                
            set(TEST_GENERATED_SOURCE_H_N "${CMAKE_BINARY_DIR}/${TARGET_NAME}/${TARGET_NAME}_${FUNDAMENTAL_TYPE}.h") #use the same name as the target	)       
            set(TEST_GENERATED_SOURCE_CPP_N "${CMAKE_BINARY_DIR}/${TARGET_NAME}/${TARGET_NAME}_${FUNDAMENTAL_TYPE}.cpp") #use the same name as the target	)       
            set(EXTENSION cpp)
            string(TOUPPER ${FUNDAMENTAL_TYPE} FUNDAMENTAL_TYPE_UPPER)
            string(TOUPPER ${EXTENSION} EXTENSION_UPPER)
            configure_file(${TEST_SOURCE_H} ${TEST_GENERATED_SOURCE_H_N} @ONLY) #replace variables in the test source file                                                                         
            configure_file(${TEST_SOURCE_CPP} ${TEST_GENERATED_SOURCE_CPP_N} @ONLY) #replace variables in the test source file                                                                         
            #message(STATUS   "Adding lib: ${TARGET_NAME}_${FUNDAMENTAL_TYPE} from ${TEST_GENERATED_SOURCE_N} and ${TEST_SOURCE}")
         
            add_generated_lib("${TARGET_NAME}_${FUNDAMENTAL_TYPE}" "${TEST_GENERATED_SOURCE_CPP_N};${TEST_GENERATED_SOURCE_H_N};" "${EXTENSION}" "${DIR_RELATIVE_PATH}")                 
            add_generated_export_header_to_target("${TARGET_NAME}_${FUNDAMENTAL_TYPE}")
 
            list(APPEND LIBS_FUNDAMENTAL_TYPE1 ${TARGET_NAME}_${FUNDAMENTAL_TYPE})
            if (CMAKE_CUDA_COMPILER AND ENABLE_CUDA) 
                set(EXTENSION cu)       
                add_cuda_to_test("${TARGET_NAME}_${FUNDAMENTAL_TYPE}")                   
                  # Process
                if(WIN32)
                    add_generated_export_header_to_target("${TARGET_NAME}_${FUNDAMENTAL_TYPE}")                      
                endif()                 
            endif()
        endforeach()
        

    set(LIBS_FUNDAMENTAL_TYPE "${LIBS_FUNDAMENTAL_TYPE1}" PARENT_SCOPE)
   #message(STATUS "Found libraries in ${DIR}: ${LIBS_FUNDAMENTAL_TYPE}")
      
endfunction()

function (discover_tests DIR)    
    file(
        GLOB_RECURSE
        TEST_SOURCES
        CONFIGURE_DEPENDS
        "${DIR}/*.h"        
    )
     
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
 