
#include <raxis/util/parallel.h>

#include <raxis/util/thread_pool.h>

#if RAXIS_PARALLEL_NATIVE

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <utility>

namespace raxis {

namespace {

// Thread-local state
thread_local bool in_parallel_region_ = false;
thread_local int thread_num_ = 0;

constexpr int NOT_SET = -1;
constexpr int CONSUMED = -2;

std::atomic<int> num_intraop_threads{NOT_SET};

int num_pool_threads(int nthreads) {
    if (nthreads == NOT_SET) {
        nthreads = intraop_default_num_threads();
    }
    // -1 because the calling thread participates as worker 0
    return nthreads - 1;
}

ThreadPool& get_intraop_pool() {
    static ThreadPool pool(
        num_pool_threads(num_intraop_threads.exchange(CONSUMED)),
        []() { raxis::init_num_threads(); });
    return pool;
}

void run_with_pool(const std::function<void(size_t)>& fn, size_t range) {
    for (size_t i = 1; i < range; ++i) {
        get_intraop_pool().run([fn, i]() { fn(i); });
    }
    // Task 0 runs on the calling thread
    fn(0);
}

struct NativeRegionGuard {
    
    explicit NativeRegionGuard(int task_id) {
        detail::set_thread_num(task_id);
        in_parallel_region_ = true;
    }
    ~NativeRegionGuard() {
        in_parallel_region_ = false;
        detail::set_thread_num(0);
    }
    NativeRegionGuard(const NativeRegionGuard&) = delete;
    NativeRegionGuard& operator=(const NativeRegionGuard&) = delete;
};

} // anonymous namespace

namespace detail {

void invoke_parallel(
    int64_t begin,
    int64_t end,
    int64_t grain_size,
    const std::function<void(int64_t, int64_t)>& f)
{
    lazy_init_num_threads();

    // Compute chunking
    int64_t total = end - begin;
    size_t num_tasks;
    int64_t chunk_size;

    if (total < grain_size) {
        num_tasks = 1;
        chunk_size = std::max(int64_t{0}, total);
    } else {
        chunk_size = divup(total, static_cast<int64_t>(raxis::get_num_threads()));
        chunk_size = std::max(grain_size, chunk_size);
        num_tasks = static_cast<size_t>(divup(total, chunk_size));
    }

    // Shared synchronisation state
    struct {
        std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
        std::exception_ptr eptr;
        std::mutex mutex;
        std::atomic_size_t remaining{0};
        std::condition_variable cv;
    } state;
    state.remaining = num_tasks;

    auto task = [&f, &state, begin, end, chunk_size](size_t task_id) {
        int64_t local_start = begin + static_cast<int64_t>(task_id) * chunk_size;
        if (local_start < end) {
            int64_t local_end = std::min(end, local_start + chunk_size);
            try {
                NativeRegionGuard guard(static_cast<int>(task_id));
                f(local_start, local_end);
            } catch (...) {
                if (!state.err_flag.test_and_set()) {
                    state.eptr = std::current_exception();
                }
            }
        }
        {
            std::unique_lock<std::mutex> lk(state.mutex);
            if (--state.remaining == 0) {
                state.cv.notify_one();
            }
        }
    };

    run_with_pool(task, num_tasks);

    // Wait for all tasks
    {
        std::unique_lock<std::mutex> lk(state.mutex);
        if (state.remaining != 0) {
            state.cv.wait(lk);
        }
    }
    if (state.eptr) std::rethrow_exception(state.eptr);
}

} // namespace internal

void init_num_threads() {
    // Nothing to do for now. The pool is created lazily.
}

void set_num_threads(int nthreads) {
    RAXIS_CHECK(nthreads > 0, "Expected positive number of threads, got ", nthreads);
    int no_value = NOT_SET;
    if (!num_intraop_threads.compare_exchange_strong(no_value, nthreads)) {
        int stored = num_intraop_threads.load();
        if (stored <= 0) {
            // Pool already created — report actual size
            stored = static_cast<int>(get_intraop_pool().size()) + 1;
        }
        RAXIS_CHECK(stored == nthreads,
            "Cannot change thread count after parallel work has started "
            "(current=", stored, ", requested=", nthreads, ")");
    }
}

int get_num_threads() {
    detail::lazy_init_num_threads();
    int n = num_intraop_threads.load();
    if (n > 0) return n;
    if (n == NOT_SET) return intraop_default_num_threads();
    // CONSUMED, pool exists
    return static_cast<int>(get_intraop_pool().size()) + 1;
}

int get_thread_num() {
    return thread_num_;
}

bool in_parallel_region() {
    return in_parallel_region_ ||
           (num_intraop_threads.load() == CONSUMED &&
            get_intraop_pool().in_thread_pool());
}

namespace detail {

void set_thread_num(int id) { thread_num_ = id; }

} // namespace detail

} // namespace raxis

#endif // RAXIS_PARALLEL_NATIVE