
#include <raxis/util/parallel.h>

#if !RAXIS_PARALLEL_OPENMP && !RAXIS_PARALLEL_NATIVE

namespace raxis {

namespace {

thread_local bool in_parallel_region_ = false;
thread_local int thread_num_ = 0;

} // anonymous namespace

namespace detail {

void set_thread_num(int thread_num) {} 

} // namespace detail


void init_num_threads() {}
void set_num_threads(int) {}
int get_num_threads() { return 1; }
int get_thread_num() { return thread_num_; }
bool in_parallel_region() { return in_parallel_region_; }

} // namespace raxis

#endif