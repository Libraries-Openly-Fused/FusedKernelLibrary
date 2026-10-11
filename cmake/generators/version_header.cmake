include(FindGit)
set(GIT_BRANCH "")
set(GIT_COMMIT_HASH "")

# git runs in FKL's directory (CMAKE_CURRENT_SOURCE_DIR: this file is included by FKL's CMakeLists.txt), which is not
# CMAKE_SOURCE_DIR when FKL is added to another project with add_subdirectory or FetchContent; outside a git checkout
# the outputs are empty
function(get_git_hash HASH)
    # Get the latest abbreviated commit hash of the working branch
    execute_process(COMMAND ${GIT_EXECUTABLE} log -1 --format=%h WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    OUTPUT_VARIABLE GIT_COMMIT_HASH OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(HASH ${GIT_COMMIT_HASH} PARENT_SCOPE)
endfunction()

function(get_git_branch BRANCH_NAME)
    execute_process(COMMAND ${GIT_EXECUTABLE} name-rev --name-only HEAD WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    OUTPUT_VARIABLE GIT_BRANCH0 OUTPUT_STRIP_TRAILING_WHITESPACE)

    string(REGEX REPLACE "((remotes/))+" " " GIT_BRANCH1 "${GIT_BRANCH0}") # remove `origin`
    string(REGEX REPLACE "((origin/))+" "" GIT_BRANCH2 "${GIT_BRANCH1}") # remove `remote`
    string(REGEX REPLACE "\\/+" "-" GIT_BRANCH3 "${GIT_BRANCH2}") # replace forward slashes with hyphens
    string(REGEX REPLACE "~[0-9]*" "" GIT_BRANCH4 "${GIT_BRANCH3}") # remove ~ followed by numbers
    string(REGEX REPLACE "\\^[0-9]*" "" GIT_BRANCH5 "${GIT_BRANCH4}") # remove ^ followed by numbers
    string(REGEX REPLACE "^-+" "" GIT_BRANCH6 "${GIT_BRANCH5}") # remove extra leading hyphens
    string(REGEX REPLACE "-+$" "" GIT_BRANCH7 "${GIT_BRANCH6}") # remove extra trailing hyphens
    string(STRIP "${GIT_BRANCH7}" GIT_BRANCH8)
    set(BRANCH_NAME ${GIT_BRANCH8} PARENT_SCOPE)
endfunction()

function(get_version_suffix VERSION_SUFFIX)
    get_git_hash(HASH)
    get_git_branch(BRANCH_NAME)
    set(VERSION_SUFFIX "${BRANCH_NAME}-${HASH}" PARENT_SCOPE)
 
endfunction()

get_version_suffix(VERSION_SUFFIX)
string(TIMESTAMP CURRENT_YEAR "%Y")

message(
    STATUS
        "${CMAKE_PROJECT_NAME} version: ${PROJECT_VERSION}-${VERSION_SUFFIX}"
)
 
