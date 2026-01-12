cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(raxis VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
option(CMAKE_INCREMENTAL_BUILD "Enable incremental builds" ON)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Runtime output directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "Debug flags:   ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "Release flags: ${CMAKE_CXX_FLAGS_RELEASE}")

# ========================= Python and site-packages ==========================
# ─────────────────────────────────────────────────────────────────────────────
# Allow to provide a custom path
set(PYTHON_EXECUTABLE "" STRING "Path to the Python executable to use")

if(PYTHON_EXECUTABLE)
    set(Python_EXECUTABLE ${PYTHON_EXECUTABLE})
endif()

# If not defined, find the current active Python environment 
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Print the result for verification
message(STATUS "Python Interpreter: ${Python_EXECUTABLE}")
message(STATUS "Python Version: ${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}")

# Get the parent directory of Python_EXECUTABLE
get_filename_component(PYTHON_EXECUTABLE_DIR ${Python_EXECUTABLE} DIRECTORY)
message(STATUS "Python executable directory: ${PYTHON_EXECUTABLE_DIR}")

if(UNIX AND NOT WIN32)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "Python site-packages directory: ${PYTHON_SITE_PACKAGES}")
    list(APPEND CMAKE_PREFIX_PATH "${PYTHON_SITE_PACKAGES}")
else()
    # Path for Windows or when using a venv
    if(WIN32)
        list(APPEND CMAKE_PREFIX_PATH "${PYTHON_EXECUTABLE_DIR}\\Lib\\site-packages")
    else()
        list(APPEND CMAKE_PREFIX_PATH 
            "${PYTHON_EXECUTABLE_DIR}/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages")
    endif()
endif()
# =============================================================================

# ============================== Raxis Library ===============================
# ─────────────────────────────────────────────────────────────────────────────
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

# Define project structure
set(RAXIS_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(RAXIS_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(RAXIS_BINDINGS_DIR ${PROJECT_SOURCE_DIR}/bindings)
set(RAXIS_EXAMPLES_DIR ${PROJECT_SOURCE_DIR}/examples)

# Collect source files (excluding bindings)
file(GLOB_RECURSE RAXIS_SOURCES "${RAXIS_SRC_DIR}/*.cpp")

# Create the core library
add_library(raxis SHARED ${RAXIS_SOURCES})

target_compile_definitions(raxis
    PRIVATE
        RAXIS_EXPORTS
)

# Set symbol visibility to hidden by default
if(NOT WIN32)
    target_compile_options(raxis PRIVATE -fvisibility=hidden)
endif()

# Set include directories
target_include_directories(raxis
    PUBLIC
        $<BUILD_INTERFACE:${RAXIS_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Link dependencies
# target_link_libraries(raxis
#     PUBLIC
#         ...
# )

add_library(raxis::raxis ALIAS raxis)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Option to build examples
option(BUILD_EXAMPLES "Build C++ Examples" ON)
if(BUILD_EXAMPLES)
    add_subdirectory("${RAXIS_EXAMPLES_DIR}")
endif()
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Option to build Python bindings
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
if(BUILD_PYTHON_BINDINGS)
    add_subdirectory("${RAXIS_BINDINGS_DIR}")

    # =============================== Installation ================================
    # ─────────────────────────────────────────────────────────────────────────────
    # 1) standard install dirs and package‐config helpers
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)
    # ─────────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────────
    # 2) install libraries and headers
    set(RAXIS_INSTALL_LIBDIR "${CMAKE_INSTALL_LIBDIR}")
    install(TARGETS raxis raxis_bindings
        EXPORT      raxisTargets
        RUNTIME     DESTINATION ${RAXIS_INSTALL_LIBDIR}   # .dll / .pyd on Windows
        LIBRARY     DESTINATION ${RAXIS_INSTALL_LIBDIR}   # .so on UNIX
        ARCHIVE     DESTINATION ${RAXIS_INSTALL_LIBDIR}   # .lib on Windows, .a on UNIX
    )

    install(DIRECTORY ${RAXIS_INCLUDE_DIR}/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
    )
    # ─────────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────────
    # 3) write out version-compat file
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/RaxisConfigVersion.cmake"
        VERSION       ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion
    )

    # 4) configure main Config file from template
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RaxisConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/RaxisConfig.cmake"
        INSTALL_DESTINATION "${CMAKE_INSTALL_DATADIR}/cmake/raxis"
    )

    # 5) install the generated Config files
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/RaxisConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/RaxisConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/cmake/raxis"
    )
    # ─────────────────────────────────────────────────────────────────────────────
endif()
# ─────────────────────────────────────────────────────────────────────────────