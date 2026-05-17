
# Helper function to build an example and copy its dependencies
function(add_example EXAMPLE_NAME SOURCE_FILE)
    add_executable(${EXAMPLE_NAME} ${SOURCE_FILE})

    set_target_properties(${EXAMPLE_NAME} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

    target_link_libraries(${EXAMPLE_NAME}
        PRIVATE
            Raxis::Raxis
    )
endfunction()

# Build each example
add_example(example_add example_add.cpp)
add_example(example_reduction example_reduction.cpp)