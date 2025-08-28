 
#shared tests libs first

set (TARGET_NAME_BASE utest_saturate)  
 
set (FUNDAMENTAL_TYPES uchar char ushort short uint int ulong long ulonglong longlong float double) 
foreach(FUNDAMENTAL_TYPE ${FUNDAMENTAL_TYPES})                  
    set(TARGET_NAME "${TARGET_NAME_BASE}_${FUNDAMENTAL_TYPE}")    
    add_generated_lib("${TARGET_NAME}" "${TARGET_NAME}.cpp;${TARGET_NAME}.h;" "/utests/algorithm/image_processing/shared")                 
    target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/exports")
    #add_generated_export_header_to_target("${TARGET_NAME}_${FUNDAMENTAL_TYPE}")
    list(APPEND LIBS_FUNDAMENTAL_TYPE ${TARGET_NAME}_${FUNDAMENTAL_TYPE})
    if (CMAKE_CUDA_COMPILER AND ENABLE_CUDA)     
        add_cuda_to_test("${TARGET_NAME}")                   
          # Process
        if(WIN32)
       #     add_generated_export_header_to_target("${TARGET_NAME}_${FUNDAMENTAL_TYPE}")                      
        endif()                 
    endif()
endforeach()  


