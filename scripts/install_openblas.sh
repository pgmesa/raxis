#!/bin/bash
# Build OpenBLAS for the aarch64 wheels, where there is no MKL. cibuildwheel
# runs it once per container (before-all); libraxis links the static library
# (FindBLAS with BLA_STATIC), so the wheel ships no libopenblas.
# From: .ci/docker/common/install_openblas.sh (PyTorch's aarch64 wheels)
#   USE_OPENMP=1: the same OpenMP runtime as libraxis (libgomp). A pthreads
#     OpenBLAS keeps its own pool spinning next to the OpenMP threads.
#   DYNAMIC_ARCH=1: kernels for every ARMv8 core, picked at runtime (Neoverse,
#     Cortex-A72 of a Raspberry Pi 4, ...), as libraxis's DispatchStub does.
# NOFORTRAN / NO_LAPACK and only single and double precision: libraxis only
# calls real CBLAS routines (PyTorch also builds complex and bfloat16).
set -e

OPENBLAS_VERSION=${OPENBLAS_VERSION:-"v0.3.34"}
PREFIX=${1:-/opt/OpenBLAS}

git clone https://github.com/OpenMathLib/OpenBLAS.git -b "${OPENBLAS_VERSION}" --depth 1 /tmp/OpenBLAS

FLAGS="USE_OPENMP=1 NUM_THREADS=256 DYNAMIC_ARCH=1 TARGET=ARMV8 NOFORTRAN=1 NO_LAPACK=1"
FLAGS="${FLAGS} BUILD_SINGLE=1 BUILD_DOUBLE=1 CFLAGS=-O3"

# One compile line per kernel, core and precision: keep them out of the CI log
echo "Building OpenBLAS ${OPENBLAS_VERSION} (log in /tmp/openblas_build.log)"
if ! make shared -j"$(nproc)" ${FLAGS} -C /tmp/OpenBLAS > /tmp/openblas_build.log 2>&1; then
    tail -n 50 /tmp/openblas_build.log
    exit 1
fi
make install ${FLAGS} PREFIX="${PREFIX}" -C /tmp/OpenBLAS > /dev/null

rm -rf /tmp/OpenBLAS
