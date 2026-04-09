
#include <raxis/util/thread_pool.h>

#include <stdexcept>


namespace raxis {

size_t ThreadPool::default_num_threads() {
    size_t n = std::thread::hardware_concurrency();
    return n == 0 ? 1 : n;
}

ThreadPool::ThreadPool(int pool_size, const std::function<void()>& init_thread)
    : threads_(pool_size < 0
                   ? static_cast<size_t>(default_num_threads())
                   : static_cast<size_t>(pool_size)),
      running_(true),
      complete_(true),
      available_(threads_.size()),
      total_(threads_.size())
{
    for (size_t i = 0; i < threads_.size(); ++i) {
        threads_[i] = std::thread([this, i, init_thread]() {
            if (init_thread) init_thread();
            this->main_loop(i);
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(mutex_);
        running_ = false;
        condition_.notify_all();
    }
    for (auto& t : threads_) {
        if (t.joinable()) t.join();
    }
}

size_t ThreadPool::size() const {
    return threads_.size();
}

size_t ThreadPool::num_available() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return available_;
}

bool ThreadPool::in_thread_pool() const {
    auto id = std::this_thread::get_id();
    for (auto& t : threads_) {
        if (t.get_id() == id) return true;
    }
    return false;
}

void ThreadPool::run(std::function<void()> func) {
    if (threads_.empty()) {
        throw std::runtime_error("ThreadPool: no threads to run a task");
    }
    std::unique_lock<std::mutex> lock(mutex_);
    tasks_.emplace(std::move(func));
    complete_ = false;
    condition_.notify_one();
}

void ThreadPool::wait_work_complete() {
    std::unique_lock<std::mutex> lock(mutex_);
    completed_.wait(lock, [&]() { return complete_; });
}

void ThreadPool::main_loop(size_t /*index*/) {
    std::unique_lock<std::mutex> lock(mutex_);
    while (running_) {
        condition_.wait(lock, [&]() { return !tasks_.empty() || !running_; });
        if (!running_) break;

        {
            TaskElement task = std::move(tasks_.front());
            tasks_.pop();
            --available_;
            lock.unlock();

            try {
                task.func();
            } catch (...) {
                // Swallow: invoke_parallel handles exceptions via its own
                // state struct. The pool itself must not die.
            }
        }

        lock.lock();
        ++available_;
        if (tasks_.empty() && available_ == total_) {
            complete_ = true;
            completed_.notify_one();
        }
    }
}

} // namespace raxis