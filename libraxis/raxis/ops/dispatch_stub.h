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
#ifdef HAVE_AVX512_CPU_DEFINITION
    AVX512 = 2,
#endif
    NUM_OPTIONS
};

RAXIS_API CPUCapability get_cpu_capability();
RAXIS_API const char* cpu_capability_name(CPUCapability cap);  

template <typename FnPtr, typename T>
struct DispatchStub;

struct RAXIS_API DispatchStubImpl {
    void* get_call_ptr(
        DeviceType device_type,
        void* DEFAULT
#ifdef HAVE_AVX2_CPU_DEFINITION
        , void* AVX2
#endif
#ifdef HAVE_AVX512_CPU_DEFINITION
        , void* AVX512
#endif
    );

    void* choose_cpu_impl(
        void* DEFAULT
#ifdef HAVE_AVX2_CPU_DEFINITION
        , void* AVX2
#endif
#ifdef HAVE_AVX512_CPU_DEFINITION
        , void* AVX512
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
#ifdef HAVE_AVX512_CPU_DEFINITION
                , reinterpret_cast<void*>(AVX512)
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
#ifdef HAVE_AVX512_CPU_DEFINITION
    static RAXIS_API FnPtr AVX512;
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

// Run when libraxis_cuda is loaded: fills the stub's CUDA slot
#define REGISTER_CUDA_DISPATCH(name, fn) \
    static RegisterCUDADispatch<struct name##_DECLARE_DISPATCH_type> name##__register(name, fn);


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

#ifdef HAVE_AVX512_CPU_DEFINITION
#define REGISTER_AVX512_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, AVX512, fn)
#else
#define REGISTER_AVX512_DISPATCH(name, fn)
#endif

#define REGISTER_ALL_CPU_DISPATCH(name, fn) \
    REGISTER_ARCH_DISPATCH(name, DEFAULT, fn) \
    REGISTER_AVX2_DISPATCH(name, fn) \
    REGISTER_AVX512_DISPATCH(name, fn)

#define REGISTER_NO_CPU_DISPATCH(name) \
    REGISTER_ALL_CPU_DISPATCH(name, nullptr)

#if defined(__CUDACC__)
#define REGISTER_DISPATCH(name, fn) REGISTER_CUDA_DISPATCH(name, fn)
#elif defined(CPU_CAPABILITY)
// From: aten/src/ATen/native/DispatchStub.h
// As in PyTorch, a kernel opts in to AVX512 with ALSO_REGISTER_AVX512_DISPATCH:
// in the AVX512 build REGISTER_DISPATCH stores nullptr and the stub falls back
// to the AVX2 kernel. Wider registers only pay off in compute-bound kernels
// (transcendentals, softmax, attention); memory-bound ones gain nothing, and
// keeping them on AVX2 keeps their summation order (and so their results)
// equal to PyTorch's.
#ifdef CPU_CAPABILITY_AVX512
#define REGISTER_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, nullptr)
#else
#define REGISTER_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, fn)
#endif
#define ALSO_REGISTER_AVX512_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, fn)
#else
#define REGISTER_DISPATCH(name, fn)
#define ALSO_REGISTER_AVX512_DISPATCH(name, fn)
#endif

} // namespace raxis