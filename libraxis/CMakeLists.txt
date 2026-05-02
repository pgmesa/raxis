cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(raxis VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(RAXIS_LIB_DIR ${PROJECT_SOURCE_DIR}/raxis)
set(RAXIS_BINDINGS_DIR ${PROJECT_SOURCE_DIR}/bindings)
set(RAXIS_EXAMPLES_DIR ${PROJECT_SOURCE_DIR}/examples)

# ─────────────────────────────────────────────────────────────────────────────
# Core library
file(GLOB_RECURSE RAXIS_SOURCES "${RAXIS_LIB_DIR}/*.cpp")
# Collect source files (excluding kernels)
list(FILTER RAXIS_SOURCES EXCLUDE REGEX ".*_kernel\\.cpp$")

add_library(raxis SHARED ${RAXIS_SOURCES})
add_library(raxis::raxis ALIAS raxis)

target_compile_definitions(raxis PRIVATE RAXIS_BUILD)

if(NOT WIN32)
    target_compile_options(raxis PRIVATE -fvisibility=hidden)
endif()

target_include_directories(raxis
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# ─────────────────────────────────────────────────────────────────────────────
# Parallelization
option(RAXIS_PARALLEL "Enable intra-op parallelism" ON)
option(RAXIS_USE_OPENMP "Prefer OpenMP over native thread pool" OFF)

if(RAXIS_PARALLEL)
    if(RAXIS_USE_OPENMP)
        find_package(OpenMP QUIET)
        if(OpenMP_CXX_FOUND)
            target_compile_definitions(raxis PUBLIC RAXIS_PARALLEL_OPENMP=1)
            target_link_libraries(raxis PUBLIC OpenMP::OpenMP_CXX)
            message(STATUS "Parallel backend: OpenMP")
        else()
            message(WARNING "OpenMP requested but not found, falling back to native")
            target_compile_definitions(raxis PUBLIC RAXIS_PARALLEL_NATIVE=1)
            message(STATUS "Parallel backend: native")
        endif()
    else()
        target_compile_definitions(raxis PUBLIC RAXIS_PARALLEL_NATIVE=1)
        message(STATUS "Parallel backend: native")
    endif()
else()
    message(STATUS "Parallel backend: serial (disabled)")
endif()

# ─────────────────────────────────────────────────────────────────────────────
# SIMD and Sleef
option(RAXIS_ENABLE_SIMD "Enable SIMD vectorization" ON)

if(RAXIS_ENABLE_SIMD)
    set(SLEEF_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(SLEEF_BUILD_DFT OFF CACHE BOOL "" FORCE)
    set(SLEEF_BUILD_GNUABI_LIBS OFF CACHE BOOL "" FORCE)
    set(SLEEF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(SLEEF_BUILD_SCALAR_LIB OFF CACHE BOOL "" FORCE)
    if(WIN32)
        set(SLEEF_BUILD_WITH_LIBM OFF CACHE BOOL "" FORCE)
    endif()
    add_subdirectory(third_party/sleef ${CMAKE_BINARY_DIR}/sleef)
    set_property(TARGET sleef PROPERTY FOLDER "dependencies")

    target_include_directories(raxis PRIVATE ${CMAKE_BINARY_DIR}/include)
    target_link_libraries(raxis PRIVATE sleef)
    target_compile_definitions(raxis PRIVATE SLEEF_STATIC_LIBS=1)
endif()

include(cmake/DispatchCompile.cmake)

file(GLOB_RECURSE RAXIS_KERNEL_SOURCES "${RAXIS_LIB_DIR}/ops/cpu/*_kernel.cpp")
raxis_add_dispatch_sources(raxis ${RAXIS_KERNEL_SOURCES})

# ─────────────────────────────────────────────────────────────────────────────
# Examples
option(BUILD_EXAMPLES "Build C++ Examples" OFF)
if(BUILD_EXAMPLES)
    add_subdirectory("${RAXIS_EXAMPLES_DIR}")
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Python bindings
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
if(BUILD_PYTHON_BINDINGS)
    add_subdirectory("${RAXIS_BINDINGS_DIR}")

    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    install(TARGETS raxis raxis_bindings
        EXPORT      raxisTargets
        RUNTIME     DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY     DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE     DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

    install(DIRECTORY ${RAXIS_LIB_DIR}/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/raxis
        FILES_MATCHING
            PATTERN "*.h"
            PATTERN "detail" EXCLUDE
    )

    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/RaxisConfigVersion.cmake"
        VERSION       ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion
    )

    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RaxisConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/RaxisConfig.cmake"
        INSTALL_DESTINATION "${CMAKE_INSTALL_DATADIR}/cmake/raxis"
    )

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/RaxisConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/RaxisConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/cmake/raxis"
    )
endif()