 

macro(add_generated_export_header_to_target TARGET_NAME)
    include(GenerateExportHeader)
    generate_export_header(${TARGET_NAME} EXPORT_FILE_NAME "${CMAKE_BINARY_DIR}/exports/${TARGET_NAME}_export.h")
    target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_BINARY_DIR}/exports/")
    set_target_properties(${TARGET_NAME} PROPERTIES CXX_VISIBILITY_PRESET default)
    target_sources(${TARGET_NAME} PRIVATE "${CMAKE_BINARY_DIR}/exports/${TARGET_NAME}_export.h")            
endmacro()
