#pragma once

#include <stdexcept>
#include <utility>

#include <raxis/export.h>
#include <raxis/core/device.h>

#include <raxis/util/exception.h>


namespace raxis {

template <typename Fn, typename Derived>
struct DispatchStub;

template <typename Ret, typename... Args, typename Derived>
struct DispatchStub<Ret(*)(Args...), Derived> {
    using fn_type = Ret(*)(Args...);

    fn_type cpu_fn = nullptr;
    fn_type cuda_fn = nullptr;

    Ret operator()(DeviceType device, Args... args) {
        switch (device) {
            case DeviceType::CPU:
                RAXIS_CHECK(cpu_fn, 
                    "Operation not implemented for ", device_type_name(device, false)); 
                return cpu_fn(std::forward<Args>(args)...);
            case DeviceType::CUDA:
                RAXIS_CHECK(cuda_fn, 
                    "Operation not implemented for ", device_type_name(device, false));
                return cuda_fn(std::forward<Args>(args)...);
            default:
                RAXIS_INTERNAL_ASSERT(false, "Unsupported device");
        }
    }
};

#define DECLARE_DISPATCH(fn_type, name)                                     \
    struct name##_dispatch_type                                              \
        : DispatchStub<fn_type, name##_dispatch_type> {};                   \
    extern RAXIS_API name##_dispatch_type name

#define DEFINE_DISPATCH(name) \
    name##_dispatch_type name

#define REGISTER_CPU_DISPATCH(stub, fn) \
    static const int stub##_cpu_reg_ = (stub.cpu_fn = (fn), 0)

#define REGISTER_CUDA_DISPATCH(stub, fn) \
    static const int stub##_cuda_reg_ = (stub.cuda_fn = (fn), 0)

} // namespace raxis