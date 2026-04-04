#pragma once

#include <memory>
#include <vector>
#include <stdexcept>
#include <cstdint>

#include <raxis/export.h>
#include <raxis/core/dtype.h>
#include <raxis/core/device.h>
#include <raxis/core/types.h>
#include <raxis/core/scalar.h>

#include <raxis/util/exception.h>

#include <raxis/tensor/detail/tensor_impl.h>
#include <raxis/tensor/tensor_options.h>


namespace raxis {

namespace indexing {

struct TensorIndex;
struct Slice;

}

class RAXIS_API Tensor {
public:
    Tensor() = default;
    explicit Tensor(std::shared_ptr<detail::TensorImpl> impl):  impl_(std::move(impl)) {};
    Tensor(const Tensor &tensor) = default;
    Tensor(Tensor &&tensor) = default;

    Tensor& operator=(const Tensor& x) & noexcept {
        impl_ = x.impl_;
        return *this;
    }
    Tensor& operator=(Tensor&& x) & noexcept {
        impl_ = std::move(x.impl_);
        return *this;
    }

    // t1[:25] = t2
    Tensor& operator=(const Tensor& rhs) && {
        return copy_(rhs);
    }

    // t1[3:5] = t3[2:4]
    Tensor& operator=(Tensor&& rhs) && {
        return copy_(rhs);
    }

    // t1[3:5] = 3
    Tensor& operator=(Scalar v) && {
        return fill_(v);
    }

    bool defined() const noexcept;
    bool is_same(const Tensor& other) const noexcept;

    const void* data_ptr() const;
    void* mutable_data_ptr() const;

    template<typename T>
    const T* data_ptr() const {
        RAXIS_CHECK(Dtype::from_type<T>() == dtype(), "Type mismatch");
        return static_cast<const T*>(data_ptr());
    }

    template<typename T>
    T* mutable_data_ptr() const {
        RAXIS_CHECK(Dtype::from_type<T>() == dtype(), "Type mismatch");
        return static_cast<T*>(mutable_data_ptr());
    }

    template<typename T>
    T item() const {
        RAXIS_CHECK(numel() == 1, 
            "a Tensor with ", numel(), " elements cannot be converted to Scalar");
        return data_ptr<T>()[0];
    }
    Scalar item() const;

    int64_t numel() const;
    int ndim() const;
    const IntArray& shape() const;
    const IntArray& strides() const;
    IntArray byte_strides() const;
    int64_t storage_offset() const;
    Dtype dtype() const;
    Device device() const;
    bool requires_grad() const;
    TensorOptions options() const;
    bool is_contiguous() const;
    
    bool is_cpu() const;
    bool is_cuda() const;

    detail::TensorImpl* unsafe_get_tensor_impl() const;
    bool is_wrapped_number() const;

    std::string to_string() const;

    // ===================== View operations ======================
    Tensor alias() const;
    Tensor select(int64_t dim, int64_t index) const;
    Tensor slice(int64_t dim, const indexing::Slice& slice) const;
    Tensor slice(int64_t dim, int64_t start, int64_t stop, int64_t step = 1) const;
    Tensor unsqueeze(int64_t dim) const;
    Tensor squeeze(std::optional<int64_t> dim_opt = std::nullopt) const;
    Tensor transpose(int64_t dim0, int64_t dim1) const;
    Tensor permute(const IntArray& dims) const;
    Tensor swapdims(int64_t dim0, int64_t dim1) const;
    Tensor swapaxes(int64_t dim0, int64_t dim1) const;
    Tensor movedim(const IntArray& src, const IntArray& dst) const;
    Tensor movedim(int64_t src, int64_t dst) const;
    Tensor view(const IntArray& shape) const;
    Tensor reshape(const IntArray& shape) const;
    Tensor broadcast_to(const IntArray& shape) const;

    // ========================= Indexing =========================
    Tensor index(const std::vector<indexing::TensorIndex>& indices) const;
    void index_put_(const std::vector<indexing::TensorIndex>& indices, const Tensor& value);
    void index_put_(const std::vector<indexing::TensorIndex>& indices, Scalar value);

    // ======================== Copy & Fill =======================
    Tensor clone() const;
    Tensor contiguous() const;
    Tensor& copy_(const Tensor& other) const;
    Tensor to(Dtype dtype) const;
    Tensor to(Device device) const;

    Tensor& fill_(Scalar value) const;
    Tensor& zero_() const;

    // =================== Reduction Operations ===================
    Tensor sum(std::optional<Dtype> dtype = std::nullopt) const;
    Tensor sum(
        const IntArray& dim, 
        bool keepdim = false, 
        std::optional<Dtype> dtype = std::nullopt) const;
        
    Tensor nansum(std::optional<Dtype> dtype = std::nullopt) const;
    Tensor nansum(
        const IntArray& dim, 
        bool keepdim = false, 
        std::optional<Dtype> dtype = std::nullopt) const;

    Tensor mean(std::optional<Dtype> dtype = std::nullopt) const;
    Tensor mean(
        const IntArray& dim, 
        bool keepdim = false, 
        std::optional<Dtype> dtype = std::nullopt) const;
        
    Tensor nanmean(std::optional<Dtype> dtype = std::nullopt) const;
    Tensor nanmean(
        const IntArray& dim, 
        bool keepdim = false, 
        std::optional<Dtype> dtype = std::nullopt) const;


    Tensor prod(std::optional<Dtype> dtype = std::nullopt) const;
    Tensor prod(
        const IntArray& dim, 
        bool keepdim = false, 
        std::optional<Dtype> dtype = std::nullopt) const;

    Tensor max() const;
    Tensor min() const;

    Tensor amax(const IntArray& dim = {}, bool keepdim = false) const;
    Tensor amin(const IntArray& dim = {}, bool keepdim = false) const;

    // ===================== Binary Operations ====================
    Tensor add(const Tensor& other) const;
    Tensor add(Scalar s) const;
    Tensor sub(const Tensor& other) const;
    Tensor sub(Scalar s) const;
    Tensor mul(const Tensor& other) const;
    Tensor mul(Scalar s) const;
    Tensor div(const Tensor& other) const;
    Tensor div(Scalar s) const;

    Tensor& add_(const Tensor& other) const;
    Tensor& add_(Scalar s) const;
    Tensor& sub_(const Tensor& other) const;
    Tensor& sub_(Scalar s) const;
    Tensor& mul_(const Tensor& other) const;
    Tensor& mul_(Scalar s) const;
    Tensor& div_(const Tensor& other) const;
    Tensor& div_(Scalar s) const;

    Tensor rsub(const Tensor& other) const;
    Tensor rsub(Scalar s) const;
    Tensor rdiv(const Tensor& other) const;
    Tensor rdiv(Scalar s) const;

    Tensor& rsub_(const Tensor& other) const;
    Tensor& rsub_(Scalar s) const;
    Tensor& rdiv_(const Tensor& other) const;
    Tensor& rdiv_(Scalar s) const;

    Tensor maximum(const Tensor& other) const;
    Tensor minimum(const Tensor& other) const;
    Tensor pow(const Tensor& exponent) const;
    Tensor pow(Scalar exponent) const;

    // =================== Comparison Operations ===================
    Tensor gt(const Tensor& other) const;
    Tensor gt(Scalar s) const;
    Tensor ge(const Tensor& other) const;
    Tensor ge(Scalar s) const;
    Tensor lt(const Tensor& other) const;
    Tensor lt(Scalar s) const;
    Tensor le(const Tensor& other) const;
    Tensor le(Scalar s) const;
    Tensor eq(const Tensor& other) const;
    Tensor eq(Scalar s) const;
    Tensor ne(const Tensor& other) const;
    Tensor ne(Scalar s) const;
    Tensor isnan() const;
 
    // ==================== Logical Operations =====================
    Tensor logical_and(const Tensor& other) const;
    Tensor logical_or(const Tensor& other) const;

    // ====================== Unary Operations =====================
    Tensor neg() const;
    Tensor abs() const;
    Tensor ceil() const;
    Tensor floor() const;
    Tensor round() const;
    Tensor exp() const;
    Tensor log() const;
    Tensor sqrt() const;
    Tensor sigmoid() const;
    Tensor logical_not() const;
    
    Tensor& neg_() const;
    Tensor& abs_() const;
    Tensor& ceil_() const;
    Tensor& floor_() const;
    Tensor& round_() const;
    Tensor& exp_() const;
    Tensor& log_() const;
    Tensor& sqrt_() const;
    Tensor& sigmoid_() const;
    Tensor& logical_not_() const;

    // Operators
    Tensor operator[](int64_t index) const;
    Tensor operator[](indexing::TensorIndex idx) const;

    Tensor operator+(const Tensor& other) const;
    Tensor operator+(Scalar s) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator-(Scalar s) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator*(Scalar s) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator/(Scalar s) const;

    Tensor& operator+=(const Tensor& other) const;
    Tensor& operator+=(Scalar s) const;
    Tensor& operator-=(const Tensor& other) const;
    Tensor& operator-=(Scalar s) const;
    Tensor& operator*=(const Tensor& other) const;
    Tensor& operator*=(Scalar s) const;
    Tensor& operator/=(const Tensor& other) const;
    Tensor& operator/=(Scalar s) const;

    // Unary
    Tensor operator-() const;
    Tensor operator~() const;

    // Comparison
    Tensor operator>(const Tensor& other) const;
    Tensor operator>(Scalar s) const;
    Tensor operator>=(const Tensor& other) const;
    Tensor operator>=(Scalar s) const;
    Tensor operator<(const Tensor& other) const;
    Tensor operator<(Scalar s) const;
    Tensor operator<=(const Tensor& other) const;
    Tensor operator<=(Scalar s) const;
    Tensor operator==(const Tensor& other) const;
    Tensor operator==(Scalar s) const;
    Tensor operator!=(const Tensor& other) const;
    Tensor operator!=(Scalar s) const;

private:
    std::shared_ptr<detail::TensorImpl> impl_;

    void ensure_defined(const char* context = nullptr) const;
};

} // namespace raxis