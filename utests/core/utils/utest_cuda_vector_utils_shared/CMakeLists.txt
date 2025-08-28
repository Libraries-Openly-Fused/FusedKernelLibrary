 
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUT_DIR})

#shared tests libs first
set (TARGET_NAME_BASE utest_cuda_vector_utils)
add_shared_test_libs(${TARGET_NAME_BASE})
 
if(${ENABLE_CPU})
    target_link_libraries(${TARGET_NAME_BASE}_cpp PRIVATE ${LIBS_FUNDAMENTAL_TYPE})
    if (MSVC)
    #   // MSVC
    #   // C4804: 'operation': unsafe use of type 'bool' in operation
    #   // C4805: 'operation': unsafe mix of type 'type' and type 'bool' in operation
    #   // C4806: 'case': unsafe case statement that forces value to bool 'true' or 'false'
       target_compile_options(${TARGET_NAME_BASE}_cpp PRIVATE  $<$<COMPILE_LANGUAGE:CXX>:/wd4804 /wd4805 /wd4806>  )
    endif()

    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        #pragma clang diagnostic ignored "-Wbool-compare"    // Compares a bool with an integer other than 0 or 1
        #pragma clang diagnostic ignored "-Wbool-operation"  // e.g., using bitwise operations on bools
        #pragma clang diagnostic ignored "-Wswitch-bool"     // e.g., switch(myBool) { case 123: ... }

        target_compile_options(${TARGET_NAME_BASE}_cpp PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wno-bool-compare>)
        target_compile_options(${TARGET_NAME_BASE}_cpp PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wno-bool-operation>)
        target_compile_options(${TARGET_NAME_BASE}_cpp PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wno-switch-bool>)
    endif()
     if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    #     // Note: GCC does not have direct equivalents for C4804 and C4806.--->but gcc and clang implement the same flags??
        target_compile_options(${TARGET_NAME_BASE}_cpp PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wno-bool-compare>)
        target_compile_options(${TARGET_NAME_BASE}_cpp PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wno-bool-operation>)
        target_compile_options(${TARGET_NAME_BASE}_cpp PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wno-switch-bool>)
    endif()
endif()


 

