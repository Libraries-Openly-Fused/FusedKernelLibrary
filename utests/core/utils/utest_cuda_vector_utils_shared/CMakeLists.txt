 
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUT_DIR})
#shared tests libs first
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/algorithm/image_processing/utest_saturate_shared)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/core/utils/utest_cuda_vector_utils_shared)

  
 
if(${ENABLE_CPU}) 
 
    target_link_libraries(utest_saturate_cpp PRIVATE ${LIBS_FUNDAMENTAL_TYPE})     
endif()

if(${ENABLE_CUDA})    
    target_link_libraries(utest_saturate_cu PRIVATE ${LIBS_FUNDAMENTAL_TYPE})
endif()

if(${ENABLE_CPU})        
    target_link_libraries(utest_cuda_vector_utils_cpp PRIVATE ${LIBS_FUNDAMENTAL_TYPE})
    if (MSVC)
    #   // MSVC
    #   // C4804: 'operation': unsafe use of type 'bool' in operation
    #   // C4805: 'operation': unsafe mix of type 'type' and type 'bool' in operation
    #   // C4806: 'case': unsafe case statement that forces value to bool 'true' or 'false'
       target_compile_options(utest_cuda_vector_utils_cpp PRIVATE  $<$<COMPILE_LANGUAGE:CXX>:/wd4804 /wd4805 /wd4806>  )
    endif()

    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        #pragma clang diagnostic ignored "-Wbool-compare"    // Compares a bool with an integer other than 0 or 1
        #pragma clang diagnostic ignored "-Wbool-operation"  // e.g., using bitwise operations on bools
        #pragma clang diagnostic ignored "-Wswitch-bool"     // e.g., switch(myBool) { case 123: ... }

        target_compile_options(utest_cuda_vector_utils_cpp PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wno-bool-compare>)
        target_compile_options(utest_cuda_vector_utils_cpp PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wno-bool-operation>)
        target_compile_options(utest_cuda_vector_utils_cpp PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wno-switch-bool>)
    endif()
     if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    #     // Note: GCC does not have direct equivalents for C4804 and C4806.--->but gcc and clang implement the same flags??
        target_compile_options(utest_cuda_vector_utils_cpp PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wno-bool-compare>)
        target_compile_options(utest_cuda_vector_utils_cpp PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wno-bool-operation>)
        target_compile_options(utest_cuda_vector_utils_cpp PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wno-switch-bool>)
    endif()
endif()


 

