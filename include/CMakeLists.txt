file(
    GLOB_RECURSE
    FKL_SOURCES
    CONFIGURE_DEPENDS    
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
)

add_library(headers INTERFACE ${FKL_SOURCES} )
# Set virtual folders for MSVC
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${FKL_SOURCES})




