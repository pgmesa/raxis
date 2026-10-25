# cmake/DispatchCompile.cmake
#
# Multi-ISA kernel compilation following PyTorch's pattern.
#
# For each kernel source (*_kernel.cpp), creates wrapper .cpp files per ISA
# that #include the real source, compiled with the appropriate flags.
#
# The RAXIS_USE_SIMD option (defined in the main CMakeLists.txt) controls
# whether SIMD variants are detected and compiled. When OFF, only DEFAULT
# is compiled. The raxis_add_dispatch_sources() function is always available.

include(CheckCXXCompilerFlag)

list(APPEND CPU_CAPABILITY_NAMES "DEFAULT")
list(APPEND CPU_CAPABILITY_FLAGS "NONE")

# Detect SIMD capabilities (only if enabled)
if(RAXIS_USE_SIMD)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|x86|X86)")
        # AVX2
        if(MSVC)
            check_cxx_compiler_flag("/arch:AVX2" CXX_AVX2_FOUND)
        else()
            check_cxx_compiler_flag("-mavx2 -mfma" CXX_AVX2_FOUND)
        endif()

        if(CXX_AVX2_FOUND)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_AVX2_CPU_DEFINITION")

            check_cxx_compiler_flag(
                "-mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store"
                COMPILER_SUPPORTS_NO_AVX256_SPLIT)
            if(COMPILER_SUPPORTS_NO_AVX256_SPLIT)
                set(CPU_NO_AVX256_SPLIT_FLAGS
                    "-mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store")
            endif()

            list(APPEND CPU_CAPABILITY_NAMES "AVX2")
            if(MSVC)
                list(APPEND CPU_CAPABILITY_FLAGS "/arch:AVX2")
            else()
                list(APPEND CPU_CAPABILITY_FLAGS "-mavx2 -mfma ${CPU_NO_AVX256_SPLIT_FLAGS}")
            endif()
            message(STATUS "Dispatch: AVX2 + FMA enabled")
        endif()

        # AVX512 (F, BW, VL, DQ), as PyTorch's cmake/Codegen.cmake
        if(MSVC)
            check_cxx_compiler_flag("/arch:AVX512" CXX_AVX512_FOUND)
        else()
            check_cxx_compiler_flag("-mavx512f -mavx512bw -mavx512vl -mavx512dq -mfma" CXX_AVX512_FOUND)
        endif()

        if(CXX_AVX2_FOUND AND CXX_AVX512_FOUND)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_AVX512_CPU_DEFINITION")
            list(APPEND CPU_CAPABILITY_NAMES "AVX512")
            if(MSVC)
                list(APPEND CPU_CAPABILITY_FLAGS "/arch:AVX512")
            else()
                list(APPEND CPU_CAPABILITY_FLAGS "-mavx512f -mavx512bw -mavx512vl -mavx512dq -mfma")
            endif()
            message(STATUS "Dispatch: AVX512 enabled")
        endif()

    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64|arm64|ARM64)")
        message(STATUS "Dispatch: AArch64 detected, NEON is baseline (DEFAULT)")
    endif()
else()
    message(STATUS "Dispatch: SIMD disabled, DEFAULT only")
endif()

list(LENGTH CPU_CAPABILITY_NAMES NUM_CPU_CAPABILITY_NAMES)
math(EXPR NUM_CPU_CAPABILITY_NAMES "${NUM_CPU_CAPABILITY_NAMES}-1")

message(STATUS "Dispatch capabilities: ${CPU_CAPABILITY_NAMES}")

function(raxis_add_dispatch_sources target)
    set(dispatch_dir "${CMAKE_CURRENT_BINARY_DIR}/dispatch")
    file(MAKE_DIRECTORY ${dispatch_dir})

    foreach(i RANGE ${NUM_CPU_CAPABILITY_NAMES})
        list(GET CPU_CAPABILITY_NAMES ${i} CPU_CAPABILITY)
        list(GET CPU_CAPABILITY_FLAGS ${i} FLAGS)

        if(FLAGS STREQUAL "NONE")
            set(FLAGS "")
        endif()

        foreach(src ${ARGN})
            get_filename_component(name ${src} NAME_WE)
            get_filename_component(abs_src ${src} ABSOLUTE)

            set(wrapper "${dispatch_dir}/${name}.${CPU_CAPABILITY}.cpp")
            set(wrapper_content "// Auto-generated: ${name} [${CPU_CAPABILITY}]\n#include \"${abs_src}\"\n")
            file(WRITE "${wrapper}.tmp" "${wrapper_content}")
            execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${wrapper}.tmp" "${wrapper}")
            file(REMOVE "${wrapper}.tmp}")

            if(MSVC)
                set(EXTRA_FLAGS "/DCPU_CAPABILITY=${CPU_CAPABILITY} /DCPU_CAPABILITY_${CPU_CAPABILITY}")
            else()
                set(EXTRA_FLAGS "-DCPU_CAPABILITY=${CPU_CAPABILITY} -DCPU_CAPABILITY_${CPU_CAPABILITY}")
            endif()

            set_source_files_properties(${wrapper} PROPERTIES
                COMPILE_FLAGS "${FLAGS} ${EXTRA_FLAGS}")
            target_sources(${target} PRIVATE ${wrapper})
        endforeach()
    endforeach()
endfunction()