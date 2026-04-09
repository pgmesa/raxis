#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace raxis {

class ThreadPool {
public:
    explicit ThreadPool(
        int pool_size, 
        const std::function<void()>& init_thread = nullptr
    );
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void run(std::function<void()> func);
    size_t size() const;
    size_t num_available() const;
    bool in_thread_pool() const;
    void wait_work_complete();

    static size_t default_num_threads();

private:
    struct TaskElement {
        std::function<void()> func;
        explicit TaskElement(std::function<void()> f) : func(std::move(f)) {}
    };

    void main_loop(size_t index);

    std::queue<TaskElement> tasks_;
    std::vector<std::thread> threads_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::condition_variable completed_;
    std::atomic_bool running_;
    bool complete_;
    size_t available_;
    size_t total_;
};

} // namespace raxis