cmake_minimum_required(VERSION 3.24)
project(xdiff LANGUAGES CXX)

add_subdirectory(external/lazex)
add_library(xdiff::lazex ALIAS lazex)

add_library(xdiff INTERFACE)
target_include_directories(xdiff INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(xdiff INTERFACE cxx_std_20)
target_link_libraries(xdiff INTERFACE lazex)

# Building xdiff itself: a target compiling each header on its own (not built by default)
# gives compile_commands.json an entry for every header.
if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    file(GLOB_RECURSE XDIFF_HEADERS CONFIGURE_DEPENDS include/*.hpp)
    target_sources(xdiff INTERFACE FILE_SET HEADERS BASE_DIRS include FILES ${XDIFF_HEADERS})
    set_target_properties(xdiff PROPERTIES VERIFY_INTERFACE_HEADER_SETS ON)
endif()
