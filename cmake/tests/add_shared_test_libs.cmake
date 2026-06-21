function (add_generated_lib TARGET_NAME TEST_SOURCES DIR)                        
        add_library(${TARGET_NAME} SHARED "${TEST_SOURCES}" )
        set_target_properties(${TARGET_NAME}  PROPERTIES LINKER_LANGUAGE CXX)      
        add_generated_export_header_to_target(${TARGET_NAME})
        configure_test_target_flags("${TARGET_NAME}" "${TEST_SOURCES}" "${DIR}")  
        set_property(TARGET "${TARGET_NAME}" PROPERTY FOLDER "${DIR}")  
endfunction()

function (add_shared_target TARGET_BASE_NAME EXTENSION FUNDAMENTAL_TYPE DIR)     
    string(TOUPPER ${FUNDAMENTAL_TYPE} FUNDAMENTAL_TYPE_UPPER)
    string(TOUPPER ${EXTENSION} EXTENSION_UPPER)
    set (GEN_DIR ${TARGET_BASE_NAME}_${FUNDAMENTAL_TYPE}_${EXTENSION})    
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_BASE_NAME}_ftype.h.in            
    ${CMAKE_BINARY_DIR}/generated/${GEN_DIR}/${GEN_DIR}.h)
    
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_BASE_NAME}_ftype.${EXTENSION}.in           
    ${CMAKE_BINARY_DIR}/generated/${GEN_DIR}/${GEN_DIR}.${EXTENSION})
 
    set(SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_BASE_NAME}_common.h" 
    ${CMAKE_BINARY_DIR}/generated/${GEN_DIR}/${GEN_DIR}.h
    ${CMAKE_BINARY_DIR}/generated/${GEN_DIR}/${GEN_DIR}.${EXTENSION})    

    add_generated_lib("${TARGET_NAME}_${EXTENSION}" "${SOURCES}"  "${DIR}")                                     
    target_include_directories("${TARGET_NAME}_${EXTENSION}" PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/")   #testcommon       
    target_include_directories("${TARGET_NAME}_${EXTENSION}" PUBLIC "${CMAKE_BINARY_DIR}/generated/${GEN_DIR}/")   #testcommon       
   
    if (MSVC)
        target_compile_options(${TARGET_NAME}_${EXTENSION} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>)
    endif()
endfunction()

function (add_shared_test_lib TARGET_BASE_NAME DIR EXTENSION FUNDAMENTAL_TYPE)
    set(TARGET_NAME "${TARGET_BASE_NAME}_${FUNDAMENTAL_TYPE}")                    
    add_shared_target("${TARGET_BASE_NAME}" "${EXTENSION}" "${FUNDAMENTAL_TYPE}" "${DIR}")         
    if ("${EXTENSION}" STREQUAL "cu")
       add_cuda_to_test("${TARGET_NAME}_${EXTENSION}")                   
    endif()                    
   
endfunction()