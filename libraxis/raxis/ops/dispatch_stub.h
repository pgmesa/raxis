#pragma once

#include <atomic>
#include <utility>

#include <raxis/export.h>
#include <raxis/core/device.h>


namespace raxis {

enum class CPUCapability {
    DEFAULT = 0,
#ifdef HAVE_AVX2_CPU_DEFINITION
    AVX2 = 1,
#endif
    NUM_OPTIONS
};

RAXIS_API CPUCapability get_cpu_capability();

template <typename FnPtr, typename T>
struct DispatchStub;

struct RAXIS_API DispatchStubImpl {
    void* get_call_ptr(
        DeviceType device_type,
        void* DEFAULT
#ifdef HAVE_AVX2_CPU_DEFINITION
        , void* AVX2
#endif
    );

    void* choose_cpu_impl(
        void* DEFAULT
#ifdef HAVE_AVX2_CPU_DEFINITION
        , void* AVX2
#endif
    );

    std::atomic<void*> cpu_dispatch_ptr{nullptr};
    void* cuda_dispatch_ptr = nullptr;
};

template <typename rT, typename T, typename... Args>
struct DispatchStub<rT (*)(Args...), T> {
    using FnPtr = rT (*)(Args...);

    DispatchStub() = default;
    DispatchStub(const DispatchStub&) = delete;
    DispatchStub& operator=(const DispatchStub&) = delete;

private:
    FnPtr get_call_ptr(DeviceType device_type) {
        return reinterpret_cast<FnPtr>(
            impl.get_call_ptr(
                device_type,
                reinterpret_cast<void*>(DEFAULT)
#ifdef HAVE_AVX2_CPU_DEFINITION
                , reinterpret_cast<void*>(AVX2)
#endif
            )
        );
    }

public:
    template <typename... ArgTypes>
    rT operator()(DeviceType device_type, ArgTypes&&... args) {
        FnPtr fn = get_call_ptr(device_type);
        return (*fn)(std::forward<ArgTypes>(args)...);
    }

    void set_cuda_dispatch_ptr(FnPtr fn_ptr) {
        impl.cuda_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
    }

    static RAXIS_API FnPtr DEFAULT;
#ifdef HAVE_AVX2_CPU_DEFINITION
    static RAXIS_API FnPtr AVX2;
#endif

private:
    DispatchStubImpl impl;
};

namespace {

template <typename DispatchStub>
struct RegisterCUDADispatch {
    RegisterCUDADispatch(DispatchStub& stub, typename DispatchStub::FnPtr value) {
        stub.set_cuda_dispatch_ptr(value);
    }
};

} // anonymous namespace


#define DECLARE_DISPATCH(fn, name)                                                         \
    struct name##_DECLARE_DISPATCH_type : DispatchStub<fn, name##_DECLARE_DISPATCH_type> {  \
        name##_DECLARE_DISPATCH_type() = default;                                          \
        name##_DECLARE_DISPATCH_type(const name##_DECLARE_DISPATCH_type&) = delete;         \
        name##_DECLARE_DISPATCH_type& operator=(const name##_DECLARE_DISPATCH_type&) = delete; \
        name##_DECLARE_DISPATCH_type(name##_DECLARE_DISPATCH_type&&) = delete;              \
        name##_DECLARE_DISPATCH_type& operator=(name##_DECLARE_DISPATCH_type&&) = delete;   \
        ~name##_DECLARE_DISPATCH_type() = default;                                         \
    };                                                                                     \
    extern RAXIS_API struct name##_DECLARE_DISPATCH_type name

#define DEFINE_DISPATCH(name) struct name##_DECLARE_DISPATCH_type name

// ---- Registration ----

#define REGISTER_ARCH_DISPATCH(name, arch, fn) \
    template <> name##_DECLARE_DISPATCH_type::FnPtr RAXIS_API \
    DispatchStub<name##_DECLARE_DISPATCH_type::FnPtr, struct name##_DECLARE_DISPATCH_type>::arch = fn;

#ifdef HAVE_AVX2_CPU_DEFINITION
#define REGISTER_AVX2_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, AVX2, fn)
#else
#define REGISTER_AVX2_DISPATCH(name, fn)
#endif

#define REGISTER_ALL_CPU_DISPATCH(name, fn) \
    REGISTER_ARCH_DISPATCH(name, DEFAULT, fn) \
    REGISTER_AVX2_DISPATCH(name, fn)

#define REGISTER_NO_CPU_DISPATCH(name) \
    REGISTER_ALL_CPU_DISPATCH(name, nullptr)

#if defined(__CUDACC__)
#define REGISTER_DISPATCH(name, fn) REGISTER_CUDA_DISPATCH(name, fn)
#elif defined(CPU_CAPABILITY)
#define REGISTER_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, fn)
#else
#define REGISTER_DISPATCH(name, fn)
#endif

} // namespace raxis