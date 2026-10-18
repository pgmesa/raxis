cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(raxis VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Needed before the target is defined: its INSTALL_INTERFACE include
# directory uses CMAKE_INSTALL_INCLUDEDIR.
include(GNUInstallDirs)

set(RAXIS_LIB_DIR ${PROJECT_SOURCE_DIR}/raxis)
set(RAXIS_BINDINGS_DIR ${PROJECT_SOURCE_DIR}/bindings)
set(RAXIS_EXAMPLES_DIR ${PROJECT_SOURCE_DIR}/examples)

# ─────────────────────────────────────────────────────────────────────────────
# Core library
file(GLOB_RECURSE RAXIS_SOURCES "${RAXIS_LIB_DIR}/*.cpp")
list(FILTER RAXIS_SOURCES EXCLUDE REGEX ".*_kernel\\.cpp$")

add_library(raxis SHARED ${RAXIS_SOURCES})
set_target_properties(raxis PROPERTIES EXPORT_NAME Raxis)
add_library(Raxis::Raxis ALIAS raxis)

target_compile_definitions(raxis PRIVATE RAXIS_BUILD)

if(NOT WIN32)
    target_compile_options(raxis PRIVATE -fvisibility=hidden)
    string(APPEND RAXIS_TARGET_CXX_FLAGS " -fvisibility=hidden")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(raxis PRIVATE -Wno-psabi)
    string(APPEND RAXIS_TARGET_CXX_FLAGS " -Wno-psabi")
endif()

target_include_directories(raxis
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# ─────────────────────────────────────────────────────────────────────────────
# BLAS
option(RAXIS_USE_BLAS "Build with BLAS library for accelerated linear algebra" ON)
option(RAXIS_USE_MKL "Prefer Intel MKL over generic BLAS (x86 only)" ON)

if(RAXIS_USE_BLAS)
    set(RAXIS_BLAS_FOUND FALSE)

    if(RAXIS_USE_MKL AND (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|x86"))
        set(MKL_LINK "static" CACHE STRING "MKL link type")
        set(MKL_INTERFACE_FULL "intel_lp64" CACHE STRING "MKL interface layer")
        set(MKL_THREADING "intel_thread" CACHE STRING "MKL threading layer")
        find_package(MKL CONFIG)
        if(MKL_FOUND)
            target_compile_definitions(raxis PRIVATE RAXIS_BUILD_WITH_BLAS=1 RAXIS_BUILD_WITH_MKL=1)
            target_link_libraries(raxis PRIVATE MKL::MKL)
            set(RAXIS_BLAS_INFO "Intel MKL (threading: ${MKL_THREADING})")
            set(RAXIS_BLAS_FOUND TRUE)

            if(WIN32)
                set(IOMP5_NAME libiomp5md.dll)
                set(IOMP5_SUFFIXES bin)
            else()
                set(IOMP5_NAME libiomp5.so)
                set(IOMP5_SUFFIXES lib lib64)
            endif()

            find_file(IOMP5_RUNTIME ${IOMP5_NAME}
                PATHS ${CMAKE_PREFIX_PATH}
                PATH_SUFFIXES ${IOMP5_SUFFIXES}
                NO_DEFAULT_PATH
            )
            if(IOMP5_RUNTIME)
                if(WIN32)
                    add_custom_command(TARGET raxis POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                            "${IOMP5_RUNTIME}"
                            "$<TARGET_FILE_DIR:raxis>"
                    )
                endif()
            else()
                message(WARNING "${IOMP5_NAME} not found, you'll need it at runtime")
            endif()
        else()
            message(STATUS "MKL: not found via CONFIG, falling back to generic BLAS")
        endif()
    elseif(RAXIS_USE_MKL)
        message(STATUS "MKL: skipped (not x86, processor: ${CMAKE_SYSTEM_PROCESSOR})")
    endif()

    if(NOT RAXIS_BLAS_FOUND)
        set(BLA_STATIC ON)
        find_package(BLAS)
        if(BLAS_FOUND)
            find_path(CBLAS_INCLUDE_DIR cblas.h
                PATH_SUFFIXES openblas openblas/include
            )
            if(CBLAS_INCLUDE_DIR)
                if(BLAS_LIBRARIES MATCHES "openblas")
                    set(RAXIS_BLAS_INFO "OpenBLAS")
                else()
                    set(RAXIS_BLAS_INFO "generic")
                endif()
                target_include_directories(raxis PRIVATE ${CBLAS_INCLUDE_DIR})
                target_link_libraries(raxis PRIVATE ${BLAS_LIBRARIES})
                target_compile_definitions(raxis PRIVATE RAXIS_BUILD_WITH_BLAS=1 RAXIS_BUILD_WITH_MKL=0)
                set(RAXIS_BLAS_FOUND TRUE)
            else()
                target_compile_definitions(raxis PRIVATE RAXIS_BUILD_WITH_BLAS=0 RAXIS_BUILD_WITH_MKL=0)
                message(WARNING "BLAS library found but cblas.h not found, using naive fallback")
            endif()
        else()
            target_compile_definitions(raxis PRIVATE RAXIS_BUILD_WITH_BLAS=0 RAXIS_BUILD_WITH_MKL=0)
            message(WARNING "BLAS requested but not found, using naive fallback kernels")
        endif()
    endif()
else()
    target_compile_definitions(raxis PRIVATE RAXIS_BUILD_WITH_BLAS=0 RAXIS_BUILD_WITH_MKL=0)
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Allocator (mimalloc)
option(RAXIS_USE_MIMALLOC "Use mimalloc for CPU memory allocation" ON)

if(RAXIS_USE_MIMALLOC)
    set(MI_OVERRIDE OFF CACHE BOOL "" FORCE)
    set(MI_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(MI_BUILD_OBJECT OFF CACHE BOOL "" FORCE)
    set(MI_BUILD_TESTS OFF CACHE BOOL "" FORCE)

    add_subdirectory(third_party/mimalloc ${CMAKE_BINARY_DIR}/mimalloc EXCLUDE_FROM_ALL)
    set_property(TARGET mimalloc-static PROPERTY FOLDER "dependencies")

    target_link_libraries(raxis PRIVATE mimalloc-static)
    target_compile_definitions(raxis PRIVATE RAXIS_USE_MIMALLOC=1)
    set(RAXIS_ALLOC_INFO "mimalloc")
else()
    set(RAXIS_ALLOC_INFO "system default")
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Parallelization
option(RAXIS_USE_PARALLEL "Enable intra-op parallelism" ON)
option(RAXIS_USE_OPENMP "Prefer OpenMP over native thread pool" ON)

if(RAXIS_USE_PARALLEL)
    if(RAXIS_BLAS_FOUND AND RAXIS_BLAS_INFO MATCHES "Intel MKL")
        target_compile_definitions(raxis PUBLIC RAXIS_PARALLEL_OPENMP=1)
        if(MSVC)
            target_compile_options(raxis PUBLIC /openmp)
            string(APPEND RAXIS_TARGET_CXX_FLAGS " /openmp")
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(raxis PUBLIC -fopenmp)
            string(APPEND RAXIS_TARGET_CXX_FLAGS " -fopenmp")
        endif()
        set(RAXIS_PARALLEL_INFO "OpenMP (via MKL/iomp5)")
    elseif(RAXIS_USE_OPENMP)
        find_package(OpenMP)
        if(OpenMP_CXX_FOUND)
            target_compile_definitions(raxis PUBLIC RAXIS_PARALLEL_OPENMP=1)
            target_link_libraries(raxis PUBLIC OpenMP::OpenMP_CXX)
            set(RAXIS_PARALLEL_INFO "OpenMP ${OpenMP_CXX_VERSION}")
        else()
            message(WARNING "OpenMP requested but not found, falling back to native")
            target_compile_definitions(raxis PUBLIC RAXIS_PARALLEL_NATIVE=1)
            set(RAXIS_PARALLEL_INFO "native thread pool")
        endif()
    else()
        target_compile_definitions(raxis PUBLIC RAXIS_PARALLEL_NATIVE=1)
        set(RAXIS_PARALLEL_INFO "native thread pool")
    endif()
else()
    set(RAXIS_PARALLEL_INFO "serial (disabled)")
endif()

# ─────────────────────────────────────────────────────────────────────────────
# SIMD and Sleef
option(RAXIS_USE_SIMD "Enable SIMD vectorization" ON)

if(RAXIS_USE_SIMD)
    set(SLEEF_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(SLEEF_BUILD_DFT OFF CACHE BOOL "" FORCE)
    set(SLEEF_BUILD_GNUABI_LIBS OFF CACHE BOOL "" FORCE)
    set(SLEEF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(SLEEF_BUILD_SCALAR_LIB OFF CACHE BOOL "" FORCE)
    if(WIN32)
        set(SLEEF_BUILD_WITH_LIBM OFF CACHE BOOL "" FORCE)
    endif()

    add_subdirectory(third_party/sleef ${CMAKE_BINARY_DIR}/sleef EXCLUDE_FROM_ALL)
    set_property(TARGET sleef PROPERTY FOLDER "dependencies")

    target_link_libraries(raxis PRIVATE sleef)
    target_compile_definitions(raxis PRIVATE SLEEF_STATIC_LIBS=1)
endif()

include(cmake/DispatchCompile.cmake)

file(GLOB_RECURSE RAXIS_KERNEL_SOURCES "${RAXIS_LIB_DIR}/ops/cpu/*_kernel.cpp")
raxis_add_dispatch_sources(raxis ${RAXIS_KERNEL_SOURCES})

# ─────────────────────────────────────────────────────────────────────────────
# Build config summary
if(NOT DEFINED RAXIS_PARALLEL_INFO)
    set(RAXIS_PARALLEL_INFO "unknown")
endif()

if(NOT DEFINED RAXIS_BLAS_INFO)
    set(RAXIS_BLAS_INFO "disabled")
endif()

if(RAXIS_USE_SIMD)
    string(REPLACE ";" ", " RAXIS_SIMD_INFO "${CPU_CAPABILITY_NAMES}")
    set(RAXIS_SLEEF_INFO "enabled")
else()
    set(RAXIS_SIMD_INFO "disabled")
    set(RAXIS_SLEEF_INFO "disabled")
endif()

if(CMAKE_BUILD_TYPE)
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _bt)
    set(RAXIS_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_bt}}")
else()
    set(RAXIS_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()
string(STRIP "${RAXIS_CXX_FLAGS}${RAXIS_TARGET_CXX_FLAGS}" RAXIS_CXX_FLAGS)

message(STATUS "")
message(STATUS "Raxis build configuration:")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  Compiler:       ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  CXX flags:      ${RAXIS_CXX_FLAGS}")
message(STATUS "  Parallel:       ${RAXIS_PARALLEL_INFO}")
message(STATUS "  SIMD dispatch:  ${RAXIS_SIMD_INFO}")
message(STATUS "  SLEEF:          ${RAXIS_SLEEF_INFO}")
message(STATUS "  BLAS:           ${RAXIS_BLAS_INFO}")
message(STATUS "  Allocator:      ${RAXIS_ALLOC_INFO}")
if(IOMP5_RUNTIME)
    message(STATUS "  iomp5 runtime:  ${IOMP5_RUNTIME}")
endif()
message(STATUS "")

configure_file(
    "${RAXIS_LIB_DIR}/util/build_config.h.in"
    "${CMAKE_BINARY_DIR}/include/raxis/util/build_config.h"
    @ONLY
)

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
            # PATTERN "detail" EXCLUDE
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

    install(
        EXPORT raxisTargets
        FILE RaxisTargets.cmake
        NAMESPACE Raxis::
        DESTINATION "${CMAKE_INSTALL_DATADIR}/cmake/raxis"
    )

    if(IOMP5_RUNTIME)
        install(FILES "${IOMP5_RUNTIME}"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )
    endif()
endif()