#pragma once

#include <raxis/util/exception.h>
#include <raxis/util/parallel_guard.h>


namespace raxis {

template <class F>
inline void parallel_for(
    int64_t begin,
    int64_t end,
    int64_t grain_size,
    const F& f)
{
    RAXIS_DEBUG_ASSERT(grain_size >= 0);
    if (begin >= end) return;

#ifdef RAXIS_INTRA_OP_PARALLEL
    detail::lazy_init_num_threads();

    const int64_t num_iters = end - begin;
    const bool use_parallel =
        (num_iters > grain_size &&
         num_iters > 1 &&
         !in_parallel_region() &&
         get_num_threads() > 1);

    if (!use_parallel) {
        detail::ThreadIdGuard tid_guard(0);
        detail::ParallelRegionGuard region_guard(true);
        f(begin, end);
        return;
    }

    detail::invoke_parallel(
        begin, end, grain_size,
        [&](int64_t chunk_begin, int64_t chunk_end) {
            detail::ParallelRegionGuard region_guard(true);
            f(chunk_begin, chunk_end);
        });
#else
    // No parallel backend, run serially
    detail::ThreadIdGuard tid_guard(0);
    detail::ParallelGuard region_guard(true);
    f(begin, end);
#endif
}


template <class scalar_t, class F, class SF>
inline scalar_t parallel_reduce(
    int64_t begin,
    int64_t end,
    int64_t grain_size,
    const scalar_t& identity,
    const F& f,
    const SF& sf)
{
    RAXIS_DEBUG_ASSERT(grain_size >= 0);
    if (begin >= end) return identity;

#ifdef RAXIS_INTRA_OP_PARALLEL
    detail::lazy_init_num_threads();

    const int max_threads = get_num_threads();
    const bool use_parallel =
        ((end - begin) > grain_size &&
         !in_parallel_region() &&
         max_threads > 1);

    if (!use_parallel) {
        detail::ThreadIdGuard tid_guard(0);
        detail::ParallelRegionGuard region_guard(true);
        return f(begin, end, identity);
    }

    std::vector<scalar_t> results(static_cast<size_t>(max_threads), identity);

    detail::invoke_parallel(
        begin, end, grain_size,
        [&](int64_t chunk_begin, int64_t chunk_end) {
            const int tid = get_thread_num();
            detail::ParallelRegionGuard region_guard(true);
            results[static_cast<size_t>(tid)] = f(chunk_begin, chunk_end, identity);
        });

    scalar_t result = identity;
    for (const auto& partial : results) {
        result = sf(result, partial);
    }
    return result;
#else
    detail::ThreadIdGuard tid_guard(0);
    detail::ParallelGuard region_guard(true);
    return f(begin, end, identity);
#endif
}

} // namespace raxis