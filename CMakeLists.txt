cmake_minimum_required(VERSION 3.14)

project(xdiff)

add_library(xdiff INTERFACE)

target_compile_features(xdiff INTERFACE cxx_std_20)

target_include_directories(xdiff
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ----------- Add interface for macros ---------------

option(XDIFF_LAZY_NESTED_DUAL "Lazy evaluation for Nested layout with runtime Nvars (avoids intermediate heap allocations)" OFF)
option(XDIFF_FAST "Aggressive inlining for Flat layout (recommended for Norder <= 3)" OFF)
option(XDIFF_LEIBNIZ_OPT "Iterative Leibniz-rule formula for Flat higher-order derivatives" OFF)
option(XDIFF_SCALAR_OPTIMIZATIONS "Optimized Dual-scalar operations for Flat layout" OFF)

target_compile_definitions(xdiff
    INTERFACE
        $<$<BOOL:${XDIFF_LAZY_NESTED_DUAL}>:XDIFF_LAZY_NESTED_DUAL>
        $<$<BOOL:${XDIFF_FAST}>:XDIFF_FAST>
        $<$<BOOL:${XDIFF_LEIBNIZ_OPT}>:XDIFF_LEIBNIZ_OPT>
        $<$<BOOL:${XDIFF_SCALAR_OPTIMIZATIONS}>:XDIFF_SCALAR_OPTIMIZATIONS>
)

# ----------- Add interface for lazy ---------------

add_subdirectory(external/lazy)

add_library(xdiff::lazy ALIAS lazy)

target_link_libraries(xdiff
    INTERFACE
        lazy
)

# ----------- Dev-only targets (benchmark) ---------------
# Only built when xdiff is configured directly (not when pulled in via
# add_subdirectory/FetchContent by another project), so consumers who only
# want the interface never build or even see this target.

if((DEFINED PROJECT_IS_TOP_LEVEL AND PROJECT_IS_TOP_LEVEL) OR
   (NOT DEFINED PROJECT_IS_TOP_LEVEL AND CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR))
    set(XDIFF_BUILD_BENCHMARK_DEFAULT ON)
else()
    set(XDIFF_BUILD_BENCHMARK_DEFAULT OFF)
endif()
option(XDIFF_BUILD_BENCHMARK "Build the xdiff benchmark executable" ${XDIFF_BUILD_BENCHMARK_DEFAULT})

if(XDIFF_BUILD_BENCHMARK)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    add_executable(benchmark benchmark.cpp)

    target_link_libraries(benchmark
        PRIVATE
            xdiff
    )

    target_compile_options(benchmark
        PRIVATE
            -O3
            -DNDEBUG
    )
endif()
