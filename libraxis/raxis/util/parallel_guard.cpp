
#include <raxis/util/parallel_guard.h>


namespace raxis::detail {

thread_local static bool in_at_parallel = false;

bool ParallelGuard::is_enabled() {
  return in_at_parallel;
}

ParallelGuard::ParallelGuard(bool state) : previous_state_(is_enabled()) {
  in_at_parallel = state;
}

ParallelGuard::~ParallelGuard() {
  in_at_parallel = previous_state_;
}
    
} // namespace raxis
