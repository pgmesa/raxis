#pragma once

#include <cstdint>
#include <algorithm>
#include <vector>

#include <raxis/export.h>
#include <raxis/util/exception.h>

namespace raxis {

// ============================================================
// Thread-count query / control
// ============================================================

// Called once per thread to lazily initialise the thread count.
RAXIS_API void init_num_threads();

// Sets the number of threads to be used in parallel region
RAXIS_API void set_num_threads(int nthreads);

// Returns the maximum number of threads that may be used in a parallel region
RAXIS_API int get_num_threads();

// Returns the current thread number (starting from 0)
// in the current parallel region, or 0 in the sequential region
RAXIS_API int get_thread_num();

// Checks whether the code runs in parallel region
RAXIS_API bool in_parallel_region();

// Returns number of intra-op threads used by default
RAXIS_API int intraop_default_num_threads();

// ============================================================
// Utilities
// ============================================================

inline int64_t divup(int64_t x, int64_t y) {
    return (x + y - 1) / y;
}

// ============================================================
// Internal
// ============================================================

namespace detail {

// Lazy, once-per-thread initialisation of the thread count.
// Uses a thread_local bool so the branch is only taken on the
// very first call from each thread.
inline void lazy_init_num_threads() {
    thread_local bool init = false;
    if (!init) [[unlikely]] {
        init_num_threads();
        init = true;
    }
}

void set_thread_num(int thread_num);

// RAII guard that saves/restores the thread-local thread id.
class ThreadIdGuard {
public:
    explicit ThreadIdGuard(int new_id) : old_id_(get_thread_num()) {
        set_thread_num(new_id);
    }

    ~ThreadIdGuard() {
        set_thread_num(old_id_);
    }

    ThreadIdGuard(const ThreadIdGuard&) = delete;
    ThreadIdGuard& operator=(const ThreadIdGuard&) = delete;

private:
    int old_id_;
};


/*
parallel_for

begin: index at which to start applying user function

end: index at which to stop applying user function

grain_size: number of elements per chunk. impacts the degree of parallelization

f: user function applied in parallel to the chunks, signature:
  void f(int64_t begin, int64_t end)

Warning: parallel_for does NOT copy thread local
states from the current thread to the worker threads.
This means for example that Tensor operations CANNOT be used in the
body of your function, only data pointers.
*/
template <class F>
inline void parallel_for(
    int64_t begin,
    int64_t end,
    int64_t grain_size,
    const F& f
);


/*
parallel_reduce

begin: index at which to start applying reduction

end: index at which to stop applying reduction

grain_size: number of elements per chunk. impacts number of elements in
intermediate results tensor and degree of parallelization.

ident: identity for binary combination function sf. sf(ident, x) needs to return
x.

f: function for reduction over a chunk. f needs to be of signature scalar_t
f(int64_t partial_begin, int64_t partial_end, scalar_t identify)

sf: function to combine two partial results. sf needs to be of signature
scalar_t sf(scalar_t x, scalar_t y)

For example, you might have a tensor of 10000 entries and want to sum together
all the elements. Parallel_reduce with a grain_size of 2500 will then allocate
an intermediate result tensor with 4 elements. Then it will execute the function
"f" you provide and pass the beginning and end index of these chunks, so
0-2499, 2500-4999, etc. and the combination identity. It will then write out
the result from each of these chunks into the intermediate result tensor. After
that it'll reduce the partial results from each chunk into a single number using
the combination function sf and the identity ident. For a total summation this
would be "+" and 0 respectively. This is similar to tbb's approach [1], where
you need to provide a function to accumulate a subrange, a function to combine
two partial results and an identity.

Warning: parallel_reduce does NOT copy thread local
states from the current thread to the worker threads.
This means for example that Tensor operations CANNOT be used in the
body of your function, only data pointers.
*/
template <class scalar_t, class F, class SF>
inline scalar_t parallel_reduce(
    int64_t begin,
    int64_t end,
    int64_t grain_size,
    const scalar_t& identity,
    const F& f,
    const SF& sf);

} // namespace detail

} // namespace raxis


// ============================================================
// Backend selection

// Detection: when OpenMP is selected AND the compiler was
// invoked with -fopenmp, _OPENMP is defined automatically.
// We use that to gate the actual parallel codegen.
// ============================================================

#if RAXIS_PARALLEL_OPENMP
#include <raxis/util/parallel_openmp.h>
#elif RAXIS_PARALLEL_NATIVE
#include <raxis/util/parallel_native.h>
#endif

#include <raxis/util/parallel_impl.h>