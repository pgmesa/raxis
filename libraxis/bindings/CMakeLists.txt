# Specify a pybind11 installation, if not, the one installed
# in the current virtual environment will be used
set(PYBIND11_PATH "" CACHE PATH "Path to Pybind11 installation")

if(PYBIND11_PATH)
    list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_PATH})
endif()

find_package(pybind11 REQUIRED)

# Create Python module
pybind11_add_module(raxis_bindings raxis_bindings.cpp)
add_library(raxis::bindings ALIAS raxis_bindings)

# Set C++ standard to match PyTorch
set_target_properties(raxis_bindings PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

# Link against core library
target_link_libraries(raxis_bindings PRIVATE raxis::raxis)

# Set output properties
set_target_properties(raxis_bindings 
    PROPERTIES
        PREFIX ""                             # no "lib" prefix
        OUTPUT_NAME "_C"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"   # .so or .pyd
        CXX_VISIBILITY_PRESET "hidden"
)