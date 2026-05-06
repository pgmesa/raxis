#include <raxis/util/parallel.h>

#if RAXIS_PARALLEL_OPENMP

#include <atomic>

#ifdef _OPENMP
#include <omp.h>
#endif

#if RAXIS_BUILD_WITH_MKL
#include <mkl.h>
#endif

namespace raxis {

namespace {
std::atomic<int> num_threads_setting{-1};
thread_local bool in_parallel_region_ = false;
thread_local int thread_num_ = 0;
} // anonymous namespace

void init_num_threads() {
    int n = num_threads_setting.load();
    if (n > 0) {
        set_num_threads(n);
    } else {
#if defined(_OPENMP) && RAXIS_BUILD_WITH_MKL
        omp_set_num_threads(mkl_get_max_threads());
#elif defined(_OPENMP)
        omp_set_num_threads(intraop_default_num_threads());
#endif
    }
}

void set_num_threads(int nthreads) {
    RAXIS_CHECK(nthreads > 0, "Expected positive number of threads, got ", nthreads);
    num_threads_setting.store(nthreads);
#ifdef _OPENMP
    omp_set_num_threads(nthreads);
#endif
#if RAXIS_BUILD_WITH_MKL
    mkl_set_num_threads_local(nthreads);
    mkl_set_dynamic(0);
#endif
}

int get_num_threads() {
#ifdef _OPENMP
    detail::lazy_init_num_threads();
    return omp_get_max_threads();
#else
    return 1;
#endif
}

int get_thread_num() {
    return thread_num_;
}

bool in_parallel_region() {
#ifdef _OPENMP
    return omp_in_parallel();
#else
    return false;
#endif
}

namespace detail {

void set_thread_num(int id) { thread_num_ = id; }

} // namespace detail

} // namespace raxis

#endif // RAXIS_PARALLEL_OPENMP