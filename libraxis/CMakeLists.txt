cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

# One version for the Python package and the C++ library (RaxisConfigVersion,
# build_config.h): the one in raxis/__init__.py
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/../raxis/__init__.py" RAXIS_VERSION_LINE
    REGEX "^__version__ = \"[0-9]+\\.[0-9]+\\.[0-9]+\"")
string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" RAXIS_VERSION "${RAXIS_VERSION_LINE}")
if(NOT RAXIS_VERSION)
    message(FATAL_ERROR "No __version__ = \"X.Y.Z\" in ../raxis/__init__.py")
endif()
project(raxis VERSION ${RAXIS_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# The static dependencies (SLEEF, mimalloc) end up inside libraxis.so. Some
# toolchains (e.g. manylinux's gcc-toolset) do not build PIC by default
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Needed before the target is defined: its INSTALL_INTERFACE include
# directory uses CMAKE_INSTALL_INCLUDEDIR.
include(GNUInstallDirs)

set(RAXIS_LIB_DIR ${PROJECT_SOURCE_DIR}/raxis)
set(RAXIS_BINDINGS_DIR ${PROJECT_SOURCE_DIR}/bindings)
set(RAXIS_EXAMPLES_DIR ${PROJECT_SOURCE_DIR}/examples)

# ─────────────────────────────────────────────────────────────────────────────
# Core library
# CONFIGURE_DEPENDS re-runs the glob on every build, so new files are picked
# up without re-configuring by hand
file(GLOB_RECURSE RAXIS_SOURCES CONFIGURE_DEPENDS "${RAXIS_LIB_DIR}/*.cpp")
list(FILTER RAXIS_SOURCES EXCLUDE REGEX ".*_kernel\\.cpp$")
# raxis/cuda and raxis/ops/cuda build libraxis_cuda (see "CUDA" below)
list(FILTER RAXIS_SOURCES EXCLUDE REGEX "/cuda/")

option(RAXIS_BUILD_STATIC "Build libraxis as a static library, for C++ projects" OFF)

if(RAXIS_BUILD_STATIC)
    if(CMAKE_VERSION VERSION_LESS 3.24)
        message(FATAL_ERROR "RAXIS_BUILD_STATIC needs CMake 3.24 or newer (WHOLE_ARCHIVE)")
    endif()
    add_library(raxis STATIC ${RAXIS_SOURCES})
    target_compile_definitions(raxis PUBLIC RAXIS_STATIC)
    set_target_properties(raxis PROPERTIES EXPORT_NAME RaxisArchive)
    # The CPU kernels (REGISTER_DISPATCH) and the CPU allocator register
    # themselves from static initializers, in object files that nothing else
    # references. A linker only takes the objects of an archive that resolve a
    # symbol, so it would drop them and every op would find an empty stub.
    # Consumers link Raxis::Raxis, which links the whole archive (libtorch's
    # static build has the same requirement)
    add_library(raxis_whole_archive INTERFACE)
    # install(EXPORT) does not rename targets inside $<LINK_LIBRARY>, so the
    # installed package names the archive by its exported name
    target_link_libraries(raxis_whole_archive INTERFACE
        "$<BUILD_INTERFACE:$<LINK_LIBRARY:WHOLE_ARCHIVE,raxis>>"
        "$<INSTALL_INTERFACE:$<LINK_LIBRARY:WHOLE_ARCHIVE,Raxis::RaxisArchive>>"
    )
    set_target_properties(raxis_whole_archive PROPERTIES EXPORT_NAME Raxis)
    add_library(Raxis::Raxis ALIAS raxis_whole_archive)
else()
    add_library(raxis SHARED ${RAXIS_SOURCES})
    set_target_properties(raxis PROPERTIES EXPORT_NAME Raxis)
    add_library(Raxis::Raxis ALIAS raxis)
endif()

# Static dependencies (SLEEF, mimalloc) go inside a shared libraxis. A static
# libraxis cannot contain them, so its consumers link them too: the target in
# the build tree, and once installed the archive installed next to libraxis
# as Raxis::<name> (RaxisConfig.cmake creates those targets)
set(RAXIS_BUNDLED_ARCHIVES "")
function(raxis_link_static_dependency dep name)
    if(RAXIS_BUILD_STATIC)
        target_link_libraries(raxis PRIVATE $<BUILD_INTERFACE:${dep}> $<INSTALL_INTERFACE:Raxis::${name}>)
        set(RAXIS_BUNDLED_ARCHIVES ${RAXIS_BUNDLED_ARCHIVES} "${dep}:${name}" PARENT_SCOPE)
    else()
        target_link_libraries(raxis PRIVATE ${dep})
    endif()
endfunction()

target_compile_definitions(raxis PRIVATE RAXIS_BUILD)

if(NOT WIN32)
    target_compile_options(raxis PRIVATE -fvisibility=hidden)
    string(APPEND RAXIS_TARGET_CXX_FLAGS " -fvisibility=hidden")
    # Installed next to its runtime dependencies (libiomp5.so) in raxis/lib.
    # macOS spells $ORIGIN as @loader_path
    if(APPLE)
        set_target_properties(raxis PROPERTIES INSTALL_RPATH "@loader_path")
    else()
        set_target_properties(raxis PROPERTIES INSTALL_RPATH "$ORIGIN")
    endif()
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
set(RAXIS_LINKS_MKL_TARGET OFF)

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
            # Read by RaxisConfig.cmake.in: a static libraxis passes MKL::MKL
            # on to its consumers, who must find MKL the same way
            set(RAXIS_LINKS_MKL_TARGET ${RAXIS_BUILD_STATIC})
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

    # Accelerate on macOS, as PyTorch's macOS wheels (BLAS=vecLib): it ships
    # with the system and its sgemm runs on Apple's matrix units. cblas.h is in
    # the vecLib framework nested inside it
    # From: cmake/Modules/FindvecLib.cmake
    if(NOT RAXIS_BLAS_FOUND AND APPLE)
        find_library(ACCELERATE_FRAMEWORK Accelerate)
        find_path(CBLAS_INCLUDE_DIR cblas.h
            PATHS "${ACCELERATE_FRAMEWORK}/Frameworks/vecLib.framework/Headers"
            NO_DEFAULT_PATH
        )
        if(ACCELERATE_FRAMEWORK AND CBLAS_INCLUDE_DIR)
            target_include_directories(raxis PRIVATE ${CBLAS_INCLUDE_DIR})
            target_link_libraries(raxis PRIVATE "-framework Accelerate")
            target_compile_definitions(raxis PRIVATE RAXIS_BUILD_WITH_BLAS=1 RAXIS_BUILD_WITH_MKL=0)
            set(RAXIS_BLAS_INFO "Accelerate")
            set(RAXIS_BLAS_FOUND TRUE)
        endif()
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
    # libraxis is loaded with dlopen (by Python): its thread locals cannot use
    # mimalloc's default initial-exec TLS model, which takes them from the
    # small static TLS block glibc reserves at startup. On aarch64 it does not
    # fit ("cannot allocate memory in static TLS block")
    set(MI_LOCAL_DYNAMIC_TLS ON CACHE BOOL "" FORCE)

    add_subdirectory(third_party/mimalloc ${CMAKE_BINARY_DIR}/mimalloc EXCLUDE_FROM_ALL)
    set_property(TARGET mimalloc-static PROPERTY FOLDER "dependencies")

    raxis_link_static_dependency(mimalloc-static mimalloc)
    target_compile_definitions(raxis PRIVATE RAXIS_USE_MIMALLOC=1)
    set(RAXIS_ALLOC_INFO "mimalloc")
else()
    set(RAXIS_ALLOC_INFO "system default")
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Parallelization
option(RAXIS_USE_PARALLEL "Enable intra-op parallelism" ON)
option(RAXIS_USE_OPENMP "Prefer OpenMP over native thread pool" ON)
# Read by RaxisConfig.cmake.in: consumers must find OpenMP before linking
set(RAXIS_LINKS_OPENMP_TARGET OFF)

if(RAXIS_USE_PARALLEL)
    if(RAXIS_BLAS_FOUND AND RAXIS_BLAS_INFO MATCHES "Intel MKL")
        target_compile_definitions(raxis PUBLIC RAXIS_PARALLEL_OPENMP=1)
        # Only for C++ sources: nvcc compiling .cu files that include raxis
        # headers rejects host compiler flags it does not know
        if(MSVC)
            target_compile_options(raxis PUBLIC $<$<COMPILE_LANGUAGE:CXX>:/openmp>)
            string(APPEND RAXIS_TARGET_CXX_FLAGS " /openmp")
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(raxis PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>)
            string(APPEND RAXIS_TARGET_CXX_FLAGS " -fopenmp")
        endif()
        set(RAXIS_PARALLEL_INFO "OpenMP (via MKL/iomp5)")
    elseif(RAXIS_USE_OPENMP)
        find_package(OpenMP)
        if(OpenMP_CXX_FOUND)
            target_compile_definitions(raxis PUBLIC RAXIS_PARALLEL_OPENMP=1)
            target_link_libraries(raxis PUBLIC OpenMP::OpenMP_CXX)
            set(RAXIS_LINKS_OPENMP_TARGET ON)
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

    raxis_link_static_dependency(sleef sleef)
    target_compile_definitions(raxis PRIVATE SLEEF_STATIC_LIBS=1)
endif()

include(cmake/DispatchCompile.cmake)

file(GLOB_RECURSE RAXIS_KERNEL_SOURCES CONFIGURE_DEPENDS "${RAXIS_LIB_DIR}/ops/cpu/*_kernel.cpp")
raxis_add_dispatch_sources(raxis ${RAXIS_KERNEL_SOURCES})

# ─────────────────────────────────────────────────────────────────────────────
# CUDA: a library of its own, as PyTorch's libtorch_cuda next to libtorch_cpu.
# libraxis does not link CUDA; libraxis_cuda links libraxis and registers its
# kernels, allocator and CUDAHooksInterface from static initializers when it
# is loaded (raxis/__init__.py loads it when present), so the CPU library and
# wheel stay the same with or without it
option(RAXIS_USE_CUDA "Build libraxis_cuda, the CUDA backend" OFF)

if(RAXIS_USE_CUDA)
    # Default archs: Turing to Blackwell as SASS, plus PTX of the newest for
    # future GPUs. An old driver cannot JIT PTX from a newer toolkit, so
    # every supported arch needs its SASS. A SASS runs on the later minors of
    # its major (sm_86 on sm_89, sm_120 on sm_121), so only the first minor
    # of each feature set is listed, as in PyTorch's wheels. The CUDAARCHS
    # environment variable (read by CMake) overrides it: the CUDA 12 wheels
    # add sm_70, which CUDA 13 dropped
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
        set(CMAKE_CUDA_ARCHITECTURES "75-real;80-real;86-real;90-real;100-real;120-real;120-virtual")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    file(GLOB_RECURSE RAXIS_CUDA_SOURCES CONFIGURE_DEPENDS
        "${RAXIS_LIB_DIR}/cuda/*.cpp"
        "${RAXIS_LIB_DIR}/cuda/*.cu"
        "${RAXIS_LIB_DIR}/ops/cuda/*.cpp"
        "${RAXIS_LIB_DIR}/ops/cuda/*.cu"
    )
    if(RAXIS_BUILD_STATIC)
        # libraxis_cuda links libraxis: with a static one it would carry its
        # own copy of the library state (dispatch tables, allocators)
        message(FATAL_ERROR "RAXIS_USE_CUDA needs a shared libraxis (RAXIS_BUILD_STATIC OFF)")
    endif()
    add_library(raxis_cuda SHARED ${RAXIS_CUDA_SOURCES})
    # The static runtime: no cudart library to ship, only the driver is needed
    # cuBLAS has no static library on Windows; the wheels will depend on
    # NVIDIA's nvidia-cublas-cu* packages, as PyTorch's
    target_link_libraries(raxis_cuda PRIVATE raxis CUDA::cudart_static CUDA::cublas)
    # The ISA macros are in CMAKE_CXX_FLAGS (cmake/DispatchCompile.cmake),
    # which nvcc does not see: without them the .cu files would compile a
    # different DispatchStub than the core
    foreach(def HAVE_AVX2_CPU_DEFINITION HAVE_AVX512_CPU_DEFINITION)
        if(CMAKE_CXX_FLAGS MATCHES "-D${def}")
            target_compile_definitions(raxis_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${def}>)
        endif()
    endforeach()
    target_compile_options(raxis_cuda PRIVATE
        # Device lambdas in host code and constexpr std:: functions
        # (std::apply, std::array) in device code, as PyTorch
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
    )
    if(MSVC)
        target_compile_options(raxis_cuda PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus>
            # nvcc's C4251 (std:: members in exported classes), which MSVC
            # only reports at higher warning levels
            $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=1394>
        )
        # cudart_static.lib asks for the static CRT (LIBCMT); everything
        # else here uses the DLL one (/MD), and two CRTs must not mix
        target_link_options(raxis_cuda PRIVATE /NODEFAULTLIB:LIBCMT)
    else()
        target_compile_options(raxis_cuda PRIVATE -fvisibility=hidden)
        # Next to libraxis (a source build), or in the raxis-cuda* wheel's
        # package, next to raxis's and the NVIDIA cuBLAS wheel's directories
        # in site-packages (raxis/__init__.py also preloads them)
        if(CUDAToolkit_VERSION_MAJOR VERSION_GREATER_EQUAL 13)
            set(RAXIS_CUBLAS_WHEEL_DIR "nvidia/cu${CUDAToolkit_VERSION_MAJOR}/lib")
        else()
            set(RAXIS_CUBLAS_WHEEL_DIR "nvidia/cublas/lib")
        endif()
        set_target_properties(raxis_cuda PROPERTIES
            INSTALL_RPATH "$ORIGIN;$ORIGIN/../../raxis/lib;$ORIGIN/../../${RAXIS_CUBLAS_WHEEL_DIR}")
    endif()
    # Read by build_libraxis.py --cuda-plugin: it names the wheel (raxis-cuda12 / raxis-cuda13)
    file(WRITE "${CMAKE_BINARY_DIR}/cuda_version.txt" "${CUDAToolkit_VERSION}\n")
    set(RAXIS_CUDA_INFO "CUDA ${CUDAToolkit_VERSION} (archs: ${CMAKE_CUDA_ARCHITECTURES})")
else()
    set(RAXIS_CUDA_INFO "disabled")
endif()

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
if(RAXIS_BUILD_STATIC)
    message(STATUS "  Library:        static")
else()
    message(STATUS "  Library:        shared")
endif()
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  CXX flags:      ${RAXIS_CXX_FLAGS}")
message(STATUS "  Parallel:       ${RAXIS_PARALLEL_INFO}")
message(STATUS "  SIMD dispatch:  ${RAXIS_SIMD_INFO}")
message(STATUS "  SLEEF:          ${RAXIS_SLEEF_INFO}")
message(STATUS "  BLAS:           ${RAXIS_BLAS_INFO}")
message(STATUS "  Allocator:      ${RAXIS_ALLOC_INFO}")
message(STATUS "  CUDA:           ${RAXIS_CUDA_INFO}")
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
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Install: the library, its headers and a CMake package (find_package(Raxis)),
# plus the Python module when it is built
include(CMakePackageConfigHelpers)

if(RAXIS_BUILD_STATIC)
    set(RAXIS_EXPORTED_TARGETS raxis raxis_whole_archive)
else()
    set(RAXIS_EXPORTED_TARGETS raxis)
endif()

install(TARGETS ${RAXIS_EXPORTED_TARGETS}
    EXPORT      raxisTargets
    RUNTIME     DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY     DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE     DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(RAXIS_USE_CUDA)
    install(TARGETS raxis_cuda
        RUNTIME     DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY     DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE     DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

if(BUILD_PYTHON_BINDINGS)
    install(TARGETS raxis_bindings
        LIBRARY     DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME     DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

install(DIRECTORY ${RAXIS_LIB_DIR}/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/raxis
    FILES_MATCHING
        PATTERN "*.h"
        # PATTERN "detail" EXCLUDE
)

# The archives of the static dependencies, under fixed names, and the
# imported targets RaxisConfig.cmake creates for them, with the system
# libraries they link themselves (mimalloc's on Windows, libm, ...)
set(RAXIS_BUNDLED_FILE "${CMAKE_CURRENT_BINARY_DIR}/RaxisBundledTargets.cmake")
file(WRITE "${RAXIS_BUNDLED_FILE}" "# Generated by libraxis/CMakeLists.txt\n")
foreach(entry ${RAXIS_BUNDLED_ARCHIVES})
    string(REPLACE ":" ";" entry "${entry}")
    list(GET entry 0 dep)
    list(GET entry 1 name)
    set(file_name "${CMAKE_STATIC_LIBRARY_PREFIX}raxis_${name}${CMAKE_STATIC_LIBRARY_SUFFIX}")
    install(FILES $<TARGET_FILE:${dep}> DESTINATION ${CMAKE_INSTALL_LIBDIR} RENAME ${file_name})
    get_target_property(dep_links ${dep} INTERFACE_LINK_LIBRARIES)
    if(NOT dep_links)
        set(dep_links "")
    endif()
    # Plain library names only: targets and generator expressions of the
    # dependency's own build do not exist for the consumer
    list(FILTER dep_links EXCLUDE REGEX "::|\$<")
    file(APPEND "${RAXIS_BUNDLED_FILE}"
        "add_library(Raxis::${name} STATIC IMPORTED)\n"
        "set_target_properties(Raxis::${name} PROPERTIES\n"
        "    IMPORTED_LOCATION \"\${_raxis_prefix}/${CMAKE_INSTALL_LIBDIR}/${file_name}\"\n"
        "    INTERFACE_LINK_LIBRARIES \"${dep_links}\")\n")
endforeach()
if(RAXIS_BUILD_STATIC)
    install(FILES "${RAXIS_BUNDLED_FILE}" DESTINATION "${CMAKE_INSTALL_DATADIR}/cmake/raxis")
endif()

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

# Ship the MSVC C++ runtime next to raxis.dll: Python only brings
# vcruntime140, not msvcp140, and an older msvcp140 found elsewhere on
# PATH can crash code built with a newer toolset
if(MSVC)
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION ${CMAKE_INSTALL_LIBDIR})
    set(CMAKE_INSTALL_UCRT_LIBRARIES OFF)
    include(InstallRequiredSystemLibraries)
endif()
