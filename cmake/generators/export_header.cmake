 

macro(add_generated_export_header_to_target TARGET_NAME BASE_NAME)
    include(GenerateExportHeader)
    string(TOUPPER ${BASE_NAME} BASE_NAME_UPPER)
    generate_export_header(${TARGET_NAME} BASE_NAME ${BASE_NAME} EXPORT_MACRO_NAME "${BASE_NAME_UPPER}_EXPORT"  EXPORT_FILE_NAME "${CMAKE_BINARY_DIR}/exports/${BASE_NAME}_export.h")    
    target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_BINARY_DIR}/exports/")
    set_target_properties(${TARGET_NAME} PROPERTIES CXX_VISIBILITY_PRESET default)
endmacro()
