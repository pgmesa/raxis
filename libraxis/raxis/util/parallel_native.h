#pragma once

#define RAXIS_INTRA_OP_PARALLEL

#include <cstdint>
#include <functional>

namespace raxis::detail {

void invoke_parallel(
    int64_t begin,
    int64_t end,
    int64_t grain_size,
    const std::function<void(int64_t, int64_t)>& f);

} // namespace raxis::detail