 
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
 

#remove shared libs, we will build targets separa

set (LIST_OF_DIRS "")
SUBDIRLIST(LIST_DIRS ${CMAKE_CURRENT_SOURCE_DIR})
 

foreach(DIR ${LIST_DIRS})
    set (ROOT_TEST_DIR "utests")
    discover_tests(${DIR})    
endforeach()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/algorithm/image_processing/utest_saturate)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/core/utils/utest_cuda_vector_utils)
 
  


 

