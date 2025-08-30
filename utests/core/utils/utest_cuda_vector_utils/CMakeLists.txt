 
#shared tests libs first
if (${ENABLE_CPU})    
    set(EXTENSION cpp)       
    string(TOUPPER ${EXTENSION} EXTENSION_UPPER)    
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/utest_cuda_vector_utils.h.in ${CMAKE_BINARY_DIR}/utest_cuda_vector_utils/utest_cuda_vector_utils_${EXTENSION}.h) #replace variables in the test source file                                                      
    add_generated_test_sub("utest_cuda_vector_utils_${EXTENSION}" "${CMAKE_BINARY_DIR}/utest_cuda_vector_utils/utest_cuda_vector_utils_${EXTENSION}.h" 
    "/utests/${EXTENSION}/core/utils/")      
endif()
add_shared_test_libs(utest_cuda_vector_utils)
 

 