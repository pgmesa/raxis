#pragma once

#include <raxis/dtype.hpp>


namespace raxis::kernels {

    void add(
        const void* a,
        const void* b,
        void* out,
        size_t n,
        const Dtype& dtype
    );

}