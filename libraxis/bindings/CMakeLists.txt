set(PYTHON_EXECUTABLE "" CACHE STRING "Path to the Python executable to use")
if(PYTHON_EXECUTABLE)
    set(Python_EXECUTABLE ${PYTHON_EXECUTABLE})
endif()

find_package(Python COMPONENTS Interpreter Development REQUIRED)

message(STATUS "Python Interpreter: ${Python_EXECUTABLE}")
message(STATUS "Python Version: ${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}")

# Ask the interpreter: deriving it from the executable path breaks on venvs
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
        "import sysconfig; print(sysconfig.get_path('purelib'))"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${PYTHON_SITE_PACKAGES}")

set(PYBIND11_PATH "" CACHE PATH "Path to pybind11 installation")
if(PYBIND11_PATH)
    list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_PATH})
endif()

find_package(pybind11 REQUIRED)
if(NOT pybind11_FOUND)
    message(FATAL_ERROR
        "pybind11 not found. Install it with: pip install pybind11")
endif()

pybind11_add_module(raxis_bindings raxis_bindings.cpp)
add_library(Raxis::Bindings ALIAS raxis_bindings)

set_target_properties(raxis_bindings PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    PREFIX ""
    OUTPUT_NAME "_C"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
    CXX_VISIBILITY_PRESET "hidden"
    INSTALL_RPATH "$ORIGIN"
)

target_link_libraries(raxis_bindings PRIVATE Raxis::Raxis)