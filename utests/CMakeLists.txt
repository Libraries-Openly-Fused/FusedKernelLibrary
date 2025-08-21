 
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
 
 
set (ROOT_TEST_DIR "utests")
discover_gen_libs("${CMAKE_SOURCE_DIR}/utests/algorithm/image_processing" "${LIBS_SATURATE}")
if(${ENABLE_CPU})
    target_include_directories(utest_saturate_cpp PRIVATE ${CMAKE_BINARY_DIR}/utest_saturate/)
    target_link_libraries(utest_saturate_cpp PRIVATE ${LIBS_SATURATE})
endif()
if(${ENABLE_CUDA})
    target_include_directories(utest_saturate_cu PRIVATE ${CMAKE_BINARY_DIR}/utest_saturate/)
    target_link_libraries(utest_saturate_cu PRIVATE ${LIBS_SATURATE})
endif()


 
