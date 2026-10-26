
#include <raxis/util/parallel.h>

#include <raxis/util/thread_pool.h>

#if RAXIS_PARALLEL_NATIVE

#include <atomic>
#include <condition_variable>
#include <memory>
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

// The pool is created on first use and replaced when set_num_threads asks
// for another size. Every task enqueue reads it, so the fast path is an
// atomic load; creating or replacing it takes the mutex
std::mutex pool_mutex;
std::unique_ptr<ThreadPool> owned_pool;
std::atomic<ThreadPool*> intraop_pool{nullptr};

std::unique_ptr<ThreadPool> make_pool(int nthreads) {
    return std::make_unique<ThreadPool>(
        num_pool_threads(nthreads), []() { raxis::init_num_threads(); });
}

ThreadPool& get_intraop_pool() {
    if (ThreadPool* pool = intraop_pool.load(std::memory_order_acquire)) {
        return *pool;
    }
    std::lock_guard<std::mutex> lock(pool_mutex);
    if (!owned_pool) {
        owned_pool = make_pool(num_intraop_threads.exchange(CONSUMED));
        intraop_pool.store(owned_pool.get(), std::memory_order_release);
    }
    return *owned_pool;
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

// From: aten/src/ATen/ParallelNative.cpp (set_num_threads)
// PyTorch's native backend only warns once the pool exists and keeps its
// size. Here an idle pool is replaced by one of the new size, so the thread
// count can change at any time, as with OpenMP (the backend of the macOS
// wheels, where Apple's clang has no OpenMP). Calling it while another
// thread runs parallel work is not supported, as in PyTorch.
void set_num_threads(int nthreads) {
    RAXIS_CHECK(nthreads > 0, "Expected positive number of threads, got ", nthreads);
    RAXIS_CHECK(!in_parallel_region(), "set_num_threads cannot be called inside a parallel region");
    std::lock_guard<std::mutex> lock(pool_mutex);
    if (!owned_pool) {
        num_intraop_threads.store(nthreads);
        return;
    }
    if (static_cast<int>(owned_pool->size()) + 1 == nthreads) {
        return;
    }
    owned_pool->wait_work_complete();
    std::unique_ptr<ThreadPool> old_pool = std::exchange(owned_pool, make_pool(nthreads));
    intraop_pool.store(owned_pool.get(), std::memory_order_release);
    // old_pool joins its idle threads here
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