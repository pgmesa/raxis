#pragma once


namespace raxis::detail {

// RAII thread local guard that tracks whether code is being executed in
// `parallel_for` or `parallel_reduce` loop function.
class ParallelGuard {
public:
    static bool is_enabled();

    ParallelGuard(bool state);
    ~ParallelGuard();

private:
    bool previous_state_;
};
    
} // namespace raxis
