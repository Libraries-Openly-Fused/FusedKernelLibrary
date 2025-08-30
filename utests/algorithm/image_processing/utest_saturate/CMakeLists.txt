 
#shared tests libs first
if (${ENABLE_CPU})    
    set(EXTENSION cpp)       
    string(TOUPPER ${EXTENSION} EXTENSION_UPPER)    
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/utest_saturate.h.in ${CMAKE_BINARY_DIR}/utest_saturate/utest_saturate_${EXTENSION}.h) #replace variables in the test source file                                                      
    add_generated_test_sub("utest_saturate_cpp" "${CMAKE_BINARY_DIR}/utest_saturate/utest_saturate_${EXTENSION}.h" 
    "/utests/${EXTENSION}/algorithm/image_processing/")      
endif()
if (CMAKE_CUDA_COMPILER AND ENABLE_CUDA)    
    set(EXTENSION cu)       
    string(TOUPPER ${EXTENSION} EXTENSION_UPPER)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/utest_saturate.h.in ${CMAKE_BINARY_DIR}/utest_saturate/utest_saturate_${EXTENSION}.h) #replace variables in the test source file                                                  
    add_generated_test_sub("utest_saturate_cu" "${CMAKE_BINARY_DIR}/utest_saturate/utest_saturate_${EXTENSION}.h" "/utests/${EXTENSION}/algorithm/image_processing/")      
    add_cuda_to_test("utest_saturate_cu")        

endif()
    add_shared_test_libs(utest_saturate)
 

 