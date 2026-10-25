#include <raxis/ops/dispatch_stub.h>

#include <raxis/core/device.h>
#include <raxis/util/exception.h>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <cpuid.h>
#endif

#include <cstdint>
#include <cstdlib>
#include <cstring>


namespace raxis {

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

namespace {

struct CpuidRegs {
    uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;
};

CpuidRegs cpuid(uint32_t leaf, uint32_t subleaf) {
    CpuidRegs r;
#if defined(_MSC_VER)
    int info[4];
    __cpuidex(info, static_cast<int>(leaf), static_cast<int>(subleaf));
    r.eax = static_cast<uint32_t>(info[0]);
    r.ebx = static_cast<uint32_t>(info[1]);
    r.ecx = static_cast<uint32_t>(info[2]);
    r.edx = static_cast<uint32_t>(info[3]);
#else
    __get_cpuid_count(leaf, subleaf, &r.eax, &r.ebx, &r.ecx, &r.edx);
#endif
    return r;
}

// XCR0: which register states the OS saves on a context switch
uint64_t xgetbv0() {
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    uint32_t lo, hi;
    asm volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return (static_cast<uint64_t>(hi) << 32) | lo;
#endif
}

} // anonymous namespace

#endif

static CPUCapability compute_cpu_capability() {
    const char* env = std::getenv("RAXIS_CPU_CAPABILITY");
    if (env) {
#ifdef HAVE_AVX512_CPU_DEFINITION
        if (std::strcmp(env, "avx512") == 0) return CPUCapability::AVX512;
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
        if (std::strcmp(env, "avx2") == 0) return CPUCapability::AVX2;
#endif
        if (std::strcmp(env, "default") == 0) return CPUCapability::DEFAULT;
    }

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) \
    && defined(HAVE_AVX2_CPU_DEFINITION)
    const CpuidRegs leaf1 = cpuid(1, 0);
    const bool os_xsave = (leaf1.ecx >> 27) & 1;
    const bool fma = (leaf1.ecx >> 12) & 1;
    if (!os_xsave || !fma) return CPUCapability::DEFAULT;

    const uint64_t xcr0 = xgetbv0();
    // XMM and YMM state (bits 1, 2)
    if ((xcr0 & 0x6) != 0x6) return CPUCapability::DEFAULT;

    const CpuidRegs leaf7 = cpuid(7, 0);

#ifdef HAVE_AVX512_CPU_DEFINITION
    // From: aten/src/ATen/native/DispatchStub.cpp (compute_cpu_capability)
    // F, DQ, BW and VL as PyTorch requires, plus the opmask and upper ZMM
    // state (XCR0 bits 5, 6, 7) saved by the OS
    const bool avx512f = (leaf7.ebx >> 16) & 1;
    const bool avx512dq = (leaf7.ebx >> 17) & 1;
    const bool avx512bw = (leaf7.ebx >> 30) & 1;
    const bool avx512vl = (leaf7.ebx >> 31) & 1;
    const bool zmm_saved = (xcr0 & 0xE0) == 0xE0;
    if (avx512f && avx512dq && avx512bw && avx512vl && zmm_saved) return CPUCapability::AVX512;
#endif

    const bool avx2 = (leaf7.ebx >> 5) & 1;
    if (avx2) return CPUCapability::AVX2;
#endif

    return CPUCapability::DEFAULT;
}

CPUCapability get_cpu_capability() {
    static CPUCapability capability = compute_cpu_capability();
    return capability;
}

const char* cpu_capability_name(CPUCapability cap) {
    switch (cap) {
#ifdef HAVE_AVX512_CPU_DEFINITION
        case CPUCapability::AVX512:
            return "AVX512";
#endif
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
#ifdef HAVE_AVX512_CPU_DEFINITION
    , void* AVX512
#endif
) {
    const CPUCapability capability = get_cpu_capability();
#ifdef HAVE_AVX512_CPU_DEFINITION
    // A kernel without an AVX512 registration runs its AVX2 version
    if (capability >= CPUCapability::AVX512 && AVX512) {
        return AVX512;
    }
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
    if (capability >= CPUCapability::AVX2 && AVX2) {
        return AVX2;
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
#ifdef HAVE_AVX512_CPU_DEFINITION
    , void* AVX512
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
#ifdef HAVE_AVX512_CPU_DEFINITION
                    , AVX512
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
