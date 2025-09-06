include (${CMAKE_SOURCE_DIR}/cmake/tests/add_shared_test_libs.cmake)
#shared tests libs first
set(TARGET_NAME utest_saturate)
if (${ENABLE_CPU})    
    set(EXTENSION cpp)       
    string(TOUPPER ${EXTENSION} EXTENSION_UPPER)    
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_NAME}.h.in 
    ${CMAKE_BINARY_DIR}/${TARGET_NAME}/${TARGET_NAME}_${EXTENSION}.h)
    
    add_generated_test_stub("${TARGET_NAME}_${EXTENSION}" 
    "${CMAKE_BINARY_DIR}/${TARGET_NAME}/${TARGET_NAME}_${EXTENSION}.h" 
    ""
    )
    set_ide_target_folder(${TARGET_NAME}_${EXTENSION} "/utests/algorithm/image_processing" ${EXTENSION})

    foreach(FUNDAMENTAL_TYPE ${FUNDAMENTAL_TYPES})   
        add_shared_test_lib(${TARGET_NAME} "" "${EXTENSION}" "${FUNDAMENTAL_TYPE}")    
        set_property(TARGET "${TARGET_NAME}_${FUNDAMENTAL_TYPE}_${EXTENSION}" PROPERTY FOLDER "/utests/algorithm/image_processing/cpu/${TARGET_NAME}")                   
        target_link_libraries(${TARGET_NAME}_${EXTENSION} PRIVATE "${TARGET_NAME}_${FUNDAMENTAL_TYPE}_${EXTENSION}")
    endforeach()

endif()

if (CMAKE_CUDA_COMPILER AND ENABLE_CUDA)    
    set(EXTENSION cu)       
    string(TOUPPER ${EXTENSION} EXTENSION_UPPER)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_NAME}.h.in 
    ${CMAKE_BINARY_DIR}/${TARGET_NAME}/${TARGET_NAME}_${EXTENSION}.h) #replace variables in the test source file                                                  
 
    add_generated_test_stub("${TARGET_NAME}_cu" 
    "${CMAKE_BINARY_DIR}/${TARGET_NAME}/${TARGET_NAME}_${EXTENSION}.h" 
    "")      
    set_ide_target_folder(${TARGET_NAME}_${EXTENSION} "/utests/algorithm/image_processing" "${EXTENSION}")
    
    add_cuda_to_test("${TARGET_NAME}_cu")    
    foreach(FUNDAMENTAL_TYPE ${FUNDAMENTAL_TYPES})   
        add_shared_test_lib(${TARGET_NAME} 
        "" 
        "${EXTENSION}"
        "${FUNDAMENTAL_TYPE}"
        )    
        set_property(TARGET "${TARGET_NAME}_${FUNDAMENTAL_TYPE}_${EXTENSION}" PROPERTY FOLDER "/utests/algorithm/image_processing/cuda/${TARGET_NAME}")                   
        target_link_libraries(${TARGET_NAME}_${EXTENSION} PRIVATE "${TARGET_NAME}_${FUNDAMENTAL_TYPE}_${EXTENSION}")
    endforeach()

endif()
  

 