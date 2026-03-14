#pragma once

#include <atomic>
#include <exception>

#ifdef _OPENMP
#define RAXIS_INTRA_OP_PARALLEL

#include <omp.h>

namespace raxis::internal {

template <typename F>
inline void invoke_parallel(
    int64_t begin, int64_t end, int64_t grain_size, const F& f)
{
    std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
    std::exception_ptr eptr;

    // Can't use omp num_threads() clause due to GOMP thread-pool bugs
    // (pytorch #32008). Instead, each thread checks whether its chunk
    // falls inside [begin, end).
#pragma omp parallel
    {
        int64_t num_threads = omp_get_num_threads();
        if (grain_size > 0) {
            num_threads = std::min(
                num_threads, raxis::divup(end - begin, grain_size));
        }

        int64_t tid = omp_get_thread_num();
        int64_t chunk_size = raxis::divup(end - begin, num_threads);
        int64_t begin_tid = begin + tid * chunk_size;

        if (begin_tid < end) {
            try {
                ThreadIdGuard tid_guard(static_cast<int>(tid));
                f(begin_tid, std::min(end, begin_tid + chunk_size));
            } catch (...) {
                if (!err_flag.test_and_set()) {
                    eptr = std::current_exception();
                }
            }
        }
    }

    if (eptr) std::rethrow_exception(eptr);
}

} // namespace raxis::internal

#endif // _OPENMP