include (${CMAKE_SOURCE_DIR}/cmake/tests/add_shared_test_libs.cmake)
if (${ENABLE_CPU})    
    set(EXTENSION cpp)    
    set(TARGET_NAME utest_cuda_vector_utils)   
    string(TOUPPER ${EXTENSION} EXTENSION_UPPER)    
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/utest_cuda_vector_utils.h.in 
    ${CMAKE_BINARY_DIR}/utest_cuda_vector_utils/utest_cuda_vector_utils_${EXTENSION}.h)
     #replace variables in the test source file   
  
    add_generated_test_stub("utest_cuda_vector_utils_${EXTENSION}"
     "${CMAKE_BINARY_DIR}/utest_cuda_vector_utils/utest_cuda_vector_utils_${EXTENSION}.h" 
    "")  
    set_ide_target_folder(${TARGET_NAME}_${EXTENSION} "/utests/core/utils" ${EXTENSION})
    foreach(FUNDAMENTAL_TYPE ${FUNDAMENTAL_TYPES})       
        add_shared_test_lib(${TARGET_NAME} "" ${EXTENSION} ${FUNDAMENTAL_TYPE})    
        set_property(TARGET "${TARGET_NAME}_${FUNDAMENTAL_TYPE}_${EXTENSION}" PROPERTY FOLDER "utests/core/utils/cpu/${TARGET_NAME}")                   
        target_link_libraries(${TARGET_NAME}_${EXTENSION} PRIVATE "${TARGET_NAME}_${FUNDAMENTAL_TYPE}_${EXTENSION}")
    endforeach()
endif()

 

 