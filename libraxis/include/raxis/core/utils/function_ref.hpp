#pragma once

namespace raxis::detail {

// Simple function_ref implementation
template <typename Signature>
class function_ref;

template <typename R, typename... Args>
class function_ref<R(Args...)> {
private:
    void* callable_;
    R (*invoker_)(void*, Args...);

public:
    template <typename F>
    function_ref(F&& f) noexcept
        : callable_(const_cast<void*>(static_cast<const void*>(&f)))
        , invoker_([](void* ptr, Args... args) -> R {
            return (*static_cast<typename std::remove_reference<F>::type*>(ptr))(
                std::forward<Args>(args)...);
        }) {}

    R operator()(Args... args) const {
        return invoker_(callable_, std::forward<Args>(args)...);
    }
};

}