
include(cmake/libs/cuda/debug.cmake)
include(cmake/libs/cuda/target_generation.cmake)
include(cmake/libs/cuda/deploy.cmake)

function(get_cuda_component_version COMPONENT COMPONENT_VERSION)
    file(READ ${CUDAToolkit_LIBRARY_ROOT}/version.json CUDA_VERSION_FILE_JSON_STRING)
    string(JSON COMPONENT_JSON_STRING GET ${CUDA_VERSION_FILE_JSON_STRING} ${IDX} ${COMPONENT})
    string(JSON COMPONENT_JSON_STRING_1 GET ${COMPONENT_JSON_STRING} ${IDX} version)
    set(${COMPONENT_VERSION} ${COMPONENT_JSON_STRING_1} PARENT_SCOPE)
endfunction()

find_package(CUDAToolkit QUIET)
if (CUDAToolkit_FOUND)
	# extra cuda_libraries only detected after project() this is needed for compatibility with old local builds that only
	# have cuda in normal location instead of custom location
	 
	# some external lbis(opencv) use findCuda, so we set this variable for compatibility
	set(CUDA_TOOLKIT_ROOT_DIR_ORIG ${CUDAToolkit_LIBRARY_ROOT})
	# file(TO_CMAKE_PATH $ENV{APIS_PATH_VS2017} APIS_PATH)
	string(REPLACE "\\" "/" CUDA_TOOLKIT_ROOT_DIR_ORIG ${CUDA_TOOLKIT_ROOT_DIR_ORIG})
	set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR_ORIG})
	message(STATUS )

	option(ENABLE_LINE_INFO "Enable line info for kernels compilation" ON)

	# Get the name from the current JSON element.
	get_cuda_component_version("cuda" CUDA_VERSION_FROM_VERSION_FILE)
	# findcudatookit requires nvcc version instead of cuda sdk version
	get_cuda_component_version("cuda_nvcc" CUDA_NVCC_VERSION_FROM_VERSION_FILE)
	 
	# split cuda version string
	string(REGEX REPLACE "([0-9]+).[0-9]+.[0-9]+" "\\1" CUDA_VERSION_MAJOR ${CUDA_VERSION_FROM_VERSION_FILE})
	string(REGEX REPLACE "[0-9]+.([0-9]+).[0-9]+" "\\1" CUDA_VERSION_MINOR ${CUDA_VERSION_FROM_VERSION_FILE})
	string(REGEX REPLACE "[0-9]+.[0-9]+.([0-9]+)" "\\1" CUDA_VERSION_REVISION ${CUDA_VERSION_FROM_VERSION_FILE})
	message(STATUS "CUDA Toolkit found. Version ${CUDA_VERSION_FROM_VERSION_FILE}")

   

else()
	message(WARNING "CUDA Toolkit not found.")
endif() 



function(add_cuda_to_target TARGET_NAME COMPONENTS)
    set_default_cuda_target_properties(${TARGET_NAME})
    # we need to deploy runtime because we se CUDA_RUNTIME_LIBRARY property to Shared
    list(APPEND COMPONENTS "cudart")
    #gpu debug code only for debug host code
    if (${ENABLE_DEBUG})    
        add_cuda_debug_support_to_target(${TARGET_NAME})
    endif()
    if (${ENABLE_NVTX})    
        add_nvtx_support_to_target(${TARGET_NAME})
    endif()
    #debug cuda code with -G already enables lineinfo, so no need to pass it
    if(${ENABLE_LINE_INFO})            
        add_cuda_lineinfo_to_target(${TARGET_NAME})
    endif()
    set(EXPORTED_CUDA_TARGETS ${COMPONENTS})
    set(COMPONENTS_TO_DEPLOY ${COMPONENTS})
    list(TRANSFORM EXPORTED_CUDA_TARGETS PREPEND "CUDA::")
    target_link_libraries(${TARGET_NAME} PRIVATE ${EXPORTED_CUDA_TARGETS})
    
    if(NOT UNIX)
        deploy_cuda_dependencies(${TARGET_NAME} "${COMPONENTS_TO_DEPLOY}")
    endif()
endfunction()
