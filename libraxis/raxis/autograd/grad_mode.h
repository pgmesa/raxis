#pragma once

#include <raxis/export.h>


namespace raxis::autograd {

struct RAXIS_API GradMode {
    static bool is_enabled();
    static void set_enabled(bool enabled);
};

// From: c10/core/GradMode.h (AutoGradMode)
// Sets grad mode for a scope and restores the previous one
struct RAXIS_API GradModeGuard {
    explicit GradModeGuard(bool enabled) : prev_mode_(GradMode::is_enabled()) {
        GradMode::set_enabled(enabled);
    }
    ~GradModeGuard() {
        GradMode::set_enabled(prev_mode_);
    }

    GradModeGuard(const GradModeGuard&) = delete;
    GradModeGuard& operator=(const GradModeGuard&) = delete;
    GradModeGuard(GradModeGuard&&) = delete;
    GradModeGuard& operator=(GradModeGuard&&) = delete;

private:
    bool prev_mode_;
};

struct RAXIS_API NoGradGuard {
    NoGradGuard() : prev_mode_(GradMode::is_enabled()) {
        GradMode::set_enabled(false);
    }
    ~NoGradGuard() {
        GradMode::set_enabled(prev_mode_);
    }

    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;
    NoGradGuard(NoGradGuard&&) = delete;
    NoGradGuard& operator=(NoGradGuard&&) = delete;

private:
    bool prev_mode_;
};

} // namespace raxis::autograd