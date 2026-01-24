
#include <raxis/factories.hpp>

#include <memory>

#include <raxis/array.hpp>
#include <raxis/array_types.hpp>
#include <raxis/core/shape.hpp>
#include <raxis/core/array_impl.hpp>
#include <raxis/core/utils/unique_void_ptr.hpp>


namespace raxis {

    NDArray from_blob(void* data, const IntArray& shape, const Dtype& dtype) {
        // Non-owning: Caller guarantees lifetime.
        auto ptr = detail::UniqueVoidPtr(data);
        auto impl = std::make_shared<ArrayImpl>(std::move(ptr), shape, dtype);
        return NDArray(std::move(impl));
    }

    NDArray from_blob(
        void* data,
        const IntArray& shape,          
        const Dtype& dtype,
        detail::DeleterFnPtr deleter
    ) {
        detail::UniqueVoidPtr ptr(data, data, deleter);
        auto impl = std::make_shared<ArrayImpl>(std::move(ptr), shape, dtype);
        return NDArray(std::move(impl));
    }

    NDArray empty(const IntArray& shape, const Dtype& dtype) {
        size_t n = static_cast<size_t>(compute_numel(shape)); //returns int64_t
        
        detail::UniqueVoidPtr ptr;

        if (n > 0) {
            void* data = std::malloc(n * dtype.itemsize());
            if (!data) {
                throw std::bad_alloc();
            }
            ptr = detail::UniqueVoidPtr(data, data, &std::free);
        }
        
        auto impl = std::make_shared<ArrayImpl>(std::move(ptr), shape, dtype);
        return NDArray(std::move(impl));
    }

    NDArray full(const IntArray& shape, double value, const Dtype& dtype) {  
        NDArray out = empty(shape, dtype);
        size_t n = static_cast<size_t>(out.numel());
        if (n > 0) dtype.fill(out.mutable_data_ptr(), n, value);
        return out;
    }

    NDArray zeros(const IntArray& shape, const Dtype& dtype) {
        return full(shape, 0, dtype);
    }

    NDArray ones(const IntArray& shape, const Dtype& dtype) {
        return full(shape, 1, dtype);
    }

}