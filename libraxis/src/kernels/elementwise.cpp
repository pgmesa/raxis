
#include <raxis/kernels/elementwise.hpp>


namespace raxis::kernels {

    template <typename T>
    void add_kernel(const void* a, const void* b, void* out, size_t n) {
        const T* ap = static_cast<const T*>(a);
        const T* bp = static_cast<const T*>(b);
        T* op = static_cast<T*>(out);

        for (size_t i = 0; i < n; ++i)
            op[i] = ap[i] + bp[i];
    }

    void add(const void* a, const void* b, void* out, size_t n, const Dtype& dtype) {
        DtypeDispatcher::dispatch(dtype, [&](auto tag) {
            using T = typename decltype(tag)::type;
            add_kernel<T>(a, b, out, n);
        });
}

} // namespace
