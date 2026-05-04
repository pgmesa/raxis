#include <raxis/ops/dispatch_stub.h>

#include <raxis/core/device.h>
#include <raxis/util/exception.h>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <cpuid.h>
#endif

#include <cstdlib>
#include <cstring>


namespace raxis {

static CPUCapability compute_cpu_capability() {
    const char* env = std::getenv("RAXIS_CPU_CAPABILITY");
    if (env) {
#ifdef HAVE_AVX2_CPU_DEFINITION
        if (std::strcmp(env, "avx2") == 0) return CPUCapability::AVX2;
#endif
        if (std::strcmp(env, "default") == 0) return CPUCapability::DEFAULT;
    }

#if defined(__aarch64__) || defined(_M_ARM64)
    return CPUCapability::DEFAULT;

#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#ifdef HAVE_AVX2_CPU_DEFINITION
#if defined(_MSC_VER)
    int info[4];

    __cpuid(info, 1);
    bool os_xsave = (info[2] >> 27) & 1;
    bool fma = (info[2] >> 12) & 1;
    if (!os_xsave || !fma) return CPUCapability::DEFAULT;

    unsigned long long xcr0 = _xgetbv(0);
    bool ymm_saved = (xcr0 & 0x6) == 0x6;
    if (!ymm_saved) return CPUCapability::DEFAULT;

    __cpuidex(info, 7, 0);
    bool avx2 = (info[1] >> 5) & 1;

    if (avx2) return CPUCapability::AVX2;
#else
    unsigned int eax, ebx, ecx, edx;

    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        return CPUCapability::DEFAULT;
    }
    bool os_xsave = (ecx >> 27) & 1;
    bool fma = (ecx >> 12) & 1;
    if (!os_xsave || !fma) return CPUCapability::DEFAULT;

    unsigned int xcr0_lo, xcr0_hi;
    asm volatile("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
    bool ymm_saved = (xcr0_lo & 0x6) == 0x6;
    if (!ymm_saved) return CPUCapability::DEFAULT;

    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return CPUCapability::DEFAULT;
    }
    bool avx2 = (ebx >> 5) & 1;

    if (avx2) return CPUCapability::AVX2;
#endif // _MSC_VER
#endif // HAVE_AVX2_CPU_DEFINITION

    return CPUCapability::DEFAULT;
#else
    return CPUCapability::DEFAULT;
#endif
}

CPUCapability get_cpu_capability() {
    static CPUCapability capability = compute_cpu_capability();
    return capability;
}

const char* cpu_capability_name(CPUCapability cap) {
    switch (cap) {
#ifdef HAVE_AVX2_CPU_DEFINITION
        case CPUCapability::AVX2:
            return "AVX2";
#endif
        case CPUCapability::DEFAULT:
            return "DEFAULT";
        default:
            return "UNKNOWN";
    }
}

void* DispatchStubImpl::choose_cpu_impl(
    void* DEFAULT
#ifdef HAVE_AVX2_CPU_DEFINITION
    , void* AVX2
#endif
) {
#ifdef HAVE_AVX2_CPU_DEFINITION
    if (get_cpu_capability() >= CPUCapability::AVX2) {
        if (AVX2) return AVX2;
    }
#endif

    RAXIS_CHECK(DEFAULT, "DispatchStub: missing default CPU kernel");
    return DEFAULT;
}

void* DispatchStubImpl::get_call_ptr(
    DeviceType device_type,
    void* DEFAULT
#ifdef HAVE_AVX2_CPU_DEFINITION
    , void* AVX2
#endif
) {
    switch (device_type) {
        case DeviceType::CPU: {
            auto fptr = cpu_dispatch_ptr.load(std::memory_order_relaxed);
            if (!fptr) {
                fptr = choose_cpu_impl(
                    DEFAULT
#ifdef HAVE_AVX2_CPU_DEFINITION
                    , AVX2
#endif
                );
                cpu_dispatch_ptr.store(fptr, std::memory_order_relaxed);
            }
            return fptr;
        }

        case DeviceType::CUDA:
            RAXIS_CHECK(cuda_dispatch_ptr, "DispatchStub: missing CUDA kernel");
            return cuda_dispatch_ptr;

        default:
            RAXIS_CHECK(false, "DispatchStub: unsupported device type");
            return nullptr;
    }
}

} // namespace raxis