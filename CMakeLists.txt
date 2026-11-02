cmake_minimum_required(VERSION 3.24)
project(xdiff LANGUAGES CXX)

add_subdirectory(external/lazex)
add_library(xdiff::lazex ALIAS lazex)

add_library(xdiff INTERFACE)
target_include_directories(xdiff INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(xdiff INTERFACE cxx_std_20)
target_link_libraries(xdiff INTERFACE lazex)

option(XDIFF_LAZY_NESTED_DUAL "Lazy evaluation for Nested layout with runtime Nvars (avoids intermediate heap allocations)" OFF)
option(XDIFF_FAST "Aggressive inlining for Flat layout (recommended for Norder <= 3)" OFF)
option(XDIFF_LEIBNIZ_OPT "Iterative Leibniz-rule formula for Flat higher-order derivatives" OFF)
option(XDIFF_SCALAR_OPTIMIZATIONS "Optimized Dual-scalar operations for Flat layout" OFF)

target_compile_definitions(xdiff INTERFACE
    $<$<BOOL:${XDIFF_LAZY_NESTED_DUAL}>:XDIFF_LAZY_NESTED_DUAL>
    $<$<BOOL:${XDIFF_FAST}>:XDIFF_FAST>
    $<$<BOOL:${XDIFF_LEIBNIZ_OPT}>:XDIFF_LEIBNIZ_OPT>
    $<$<BOOL:${XDIFF_SCALAR_OPTIMIZATIONS}>:XDIFF_SCALAR_OPTIMIZATIONS>
)

# Building xdiff itself: a target compiling each header on its own (not built by default)
# gives compile_commands.json an entry for every header.
if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    file(GLOB_RECURSE XDIFF_HEADERS CONFIGURE_DEPENDS include/*.hpp)
    target_sources(xdiff INTERFACE FILE_SET HEADERS BASE_DIRS include FILES ${XDIFF_HEADERS})
    set_target_properties(xdiff PROPERTIES VERIFY_INTERFACE_HEADER_SETS ON)
endif()
