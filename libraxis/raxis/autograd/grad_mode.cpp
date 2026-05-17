
#include <raxis/autograd/grad_mode.h>


namespace raxis::autograd {

static thread_local bool grad_mode_enabled = true;

bool GradMode::is_enabled() {
    return grad_mode_enabled;
}

void GradMode::set_enabled(bool enabled) {
    grad_mode_enabled = enabled;
}

} // namespace raxis::autograd