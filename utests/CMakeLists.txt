 
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUT_DIR})

MACRO(SUBDIRLIST result curdir)
FILE(GLOB children ${curdir}/*) # 
SET(dirlist "")
FOREACH(child ${children})
    IF(IS_DIRECTORY ${child}) 
        LIST(APPEND dirlist ${child})
    ENDIF()
ENDFOREACH()
SET(${result} ${dirlist} )
ENDMACRO()
 
set (LIST_OF_DIRS "")
SUBDIRLIST(LIST_DIRS ${CMAKE_CURRENT_SOURCE_DIR})


foreach(DIR ${LIST_DIRS})
    set (ROOT_TEST_DIR "utests")
    discover_tests(${DIR})    
endforeach()

set (LIBS_SATURATE "")
set (LIBS_VECTOR_UTILS "")
  
if(${ENABLE_CPU}) 
    discover_gen_libs("${CMAKE_SOURCE_DIR}/utests/algorithm/image_processing" "${LIBS_SATURATE}")
    target_include_directories(utest_saturate_cpp PRIVATE ${CMAKE_BINARY_DIR}/utest_saturate/)
    target_link_libraries(utest_saturate_cpp PRIVATE ${LIBS_FUNDAMENTAL_TYPE})     
endif()

if(${ENABLE_CUDA})
    target_include_directories(utest_saturate_cu PRIVATE ${CMAKE_BINARY_DIR}/utest_saturate/)
    target_link_libraries(utest_saturate_cu PRIVATE ${LIBS_FUNDAMENTAL_TYPE})
endif()

if(${ENABLE_CPU})
    discover_gen_libs("${CMAKE_SOURCE_DIR}/utests/core/utils" "${LIBS_FUNDAMENTAL_TYPE}")
    target_include_directories(utest_cuda_vector_utils_cpp PRIVATE ${CMAKE_BINARY_DIR}/utest_cuda_vector_utils/)
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


 

