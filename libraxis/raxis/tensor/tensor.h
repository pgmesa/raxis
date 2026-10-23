#pragma once

#include <memory>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <optional>

#include <raxis/export.h>
#include <raxis/core/dtype.h>
#include <raxis/core/device.h>
#include <raxis/core/types.h>
#include <raxis/core/scalar.h>
#include <raxis/core/memory_format.h>

#include <raxis/util/exception.h>

#include <raxis/tensor/detail/tensor_impl.h>
#include <raxis/tensor/tensor_options.h>


namespace raxis {

namespace indexing {
struct TensorIndex;
struct Slice;
} // namespace indexing

namespace autograd { 
struct AutogradMeta;
class Node; 
} // namespace autograd

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
    size_t nbytes() const;
    int ndim() const;
    IntArrayRef shape() const;
    int64_t stride(int64_t dim) const;
    IntArrayRef strides() const;
    DimVector byte_strides() const;
    int64_t storage_offset() const;
    Dtype dtype() const;
    Device device() const;
    TensorOptions options() const;
    bool is_contiguous() const;
    bool is_non_overlapping_and_dense() const;
    
    bool is_cpu() const;
    bool is_cuda() const;

    detail::TensorImpl* unsafe_get_tensor_impl() const;
    // Non-owning handle for structures that must not keep the tensor alive
    std::weak_ptr<detail::TensorImpl> weak_impl() const { return impl_; }
    const std::shared_ptr<detail::TensorImpl>& impl_ptr() const { return impl_; }
    long use_count() const noexcept { return impl_.use_count(); }
    bool is_wrapped_number() const;

    // Values, type and shape as printed by operator<< (tensor/formatting.h)
    std::string to_string() const;

    // ======================== Autograd =========================
    bool requires_grad() const;
    Tensor& requires_grad_(bool requires_grad = true);

    void retain_grad() const;
    bool retains_grad() const;
    
    Tensor grad() const;
    void set_grad(const Tensor& grad);
    
    std::shared_ptr<autograd::Node> grad_fn() const;
    void set_grad_fn(std::shared_ptr<autograd::Node> fn);
    
    uint32_t output_nr() const;
    void set_output_nr(uint32_t nr);
    
    void set_requires_grad(bool requires_grad);
    
    bool is_leaf() const;
    
    Tensor detach() const;

    // Version of the memory: every in-place op on this tensor (or on a
    // tensor sharing its memory) increments it
    uint32_t version() const;

    // Differentiable views: the tensor whose memory they share (undefined
    // for a tensor that is not a view)
    bool is_view() const;
    Tensor base() const;
    
    // retain_graph keeps the saved tensors for another backward;
    // create_graph records the backward, so the gradients can be
    // differentiated again (it implies retain_graph)
    void backward(const Tensor& grad = {}, bool retain_graph = false, bool create_graph = false) const;

    // ===================== View operations ======================
    Tensor alias() const;
    Tensor select(int64_t dim, int64_t index) const;
    Tensor slice(int64_t dim, const indexing::Slice& slice) const;
    Tensor slice(int64_t dim, int64_t start, int64_t stop, int64_t step = 1) const;
    Tensor narrow(int64_t dim, int64_t start, int64_t length) const;
    Tensor unsqueeze(int64_t dim) const;
    Tensor squeeze(std::optional<int64_t> dim_opt = std::nullopt) const;
    Tensor mT() const;
    Tensor transpose(int64_t dim0, int64_t dim1) const;
    Tensor permute(IntArrayRef dims) const;
    Tensor swapdims(int64_t dim0, int64_t dim1) const;
    Tensor movedim(IntArrayRef src, IntArrayRef dst) const;
    Tensor movedim(int64_t src, int64_t dst) const;
    Tensor view(IntArrayRef shape) const;
    Tensor reshape(IntArrayRef shape) const;
    Tensor flatten(int64_t start_dim = 0, int64_t end_dim = -1) const;
    Tensor expand(IntArrayRef shape) const;
    Tensor broadcast_to(IntArrayRef shape) const;

    // ========================= Indexing =========================
    Tensor index(ArrayRef<indexing::TensorIndex> indices) const;
    void index_put_(ArrayRef<indexing::TensorIndex> indices, const Tensor& value);
    void index_put_(ArrayRef<indexing::TensorIndex> indices, Scalar value);

    // ======================== Copy & Fill =======================
    Tensor clone(MemoryFormat memory_format = MemoryFormat::Preserve) const;
    Tensor contiguous() const;
    Tensor& copy_(const Tensor& other) const;
    Tensor to(Dtype dtype) const;
    Tensor to(Device device) const;
    Tensor cpu() const;

    Tensor& fill_(Scalar value) const;
    Tensor& zero_() const;
    Tensor& uniform_(double low = 0.0, double high = 1.0) const;
    Tensor& normal_(double mean = 0.0, double std = 1.0) const;
    Tensor& randint_(int64_t low, int64_t high) const;

    // =================== Reduction Operations ===================
    Tensor sum(std::optional<Dtype> dtype = std::nullopt) const;
    Tensor sum(
        IntArrayRef dim, 
        bool keepdim = false, 
        std::optional<Dtype> dtype = std::nullopt) const;
        
    Tensor nansum(std::optional<Dtype> dtype = std::nullopt) const;
    Tensor nansum(
        IntArrayRef dim, 
        bool keepdim = false, 
        std::optional<Dtype> dtype = std::nullopt) const;

    Tensor mean(std::optional<Dtype> dtype = std::nullopt) const;
    Tensor mean(
        IntArrayRef dim, 
        bool keepdim = false, 
        std::optional<Dtype> dtype = std::nullopt) const;
        
    Tensor nanmean(std::optional<Dtype> dtype = std::nullopt) const;
    Tensor nanmean(
        IntArrayRef dim, 
        bool keepdim = false, 
        std::optional<Dtype> dtype = std::nullopt) const;


    Tensor prod(std::optional<Dtype> dtype = std::nullopt) const;
    Tensor prod(
        IntArrayRef dim, 
        bool keepdim = false, 
        std::optional<Dtype> dtype = std::nullopt) const;

    Tensor max() const;
    Tensor min() const;

    Tensor amax(IntArrayRef dim = {}, bool keepdim = false) const;
    Tensor amin(IntArrayRef dim = {}, bool keepdim = false) const;

    Tensor cumsum(int64_t dim, std::optional<Dtype> dtype = std::nullopt) const;

    Tensor all(IntArrayRef dim = {}, bool keepdim = false) const;
    Tensor any(IntArrayRef dim = {}, bool keepdim = false) const;

    // ====================== Linear Algebra ======================
    Tensor dot(const Tensor& other) const;
    Tensor mv(const Tensor& vec) const;
    Tensor mm(const Tensor& mat2) const;
    Tensor bmm(const Tensor& mat2) const;
    Tensor matmul(const Tensor& other) const;

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
    Tensor sign() const;
    Tensor ceil() const;
    Tensor floor() const;
    Tensor round() const;
    Tensor exp() const;
    Tensor log() const;
    Tensor sqrt() const;
    Tensor rsqrt() const;
    Tensor reciprocal() const;
    Tensor sigmoid() const;
    Tensor relu() const;
    Tensor softmax(int64_t dim) const;
    Tensor log_softmax(int64_t dim) const;
    Tensor logical_not() const;
    Tensor expm1() const;
    Tensor log2() const;
    Tensor log10() const;
    Tensor log1p() const;
    Tensor sin() const;
    Tensor cos() const;
    Tensor tan() const;
    Tensor asin() const;
    Tensor acos() const;
    Tensor atan() const;
    Tensor sinh() const;
    Tensor cosh() const;
    Tensor tanh() const;
    Tensor erf() const;
    Tensor erfc() const;
    
    Tensor& neg_() const;
    Tensor& abs_() const;
    Tensor& sign_() const;
    Tensor& ceil_() const;
    Tensor& floor_() const;
    Tensor& round_() const;
    Tensor& exp_() const;
    Tensor& log_() const;
    Tensor& sqrt_() const;
    Tensor& rsqrt_() const;
    Tensor& reciprocal_() const;
    Tensor& sigmoid_() const;
    Tensor& relu_() const;
    Tensor& logical_not_() const;
    Tensor& expm1_() const;
    Tensor& log2_() const;
    Tensor& log10_() const;
    Tensor& log1p_() const;
    Tensor& sin_() const;
    Tensor& cos_() const;
    Tensor& tan_() const;
    Tensor& asin_() const;
    Tensor& acos_() const;
    Tensor& atan_() const;
    Tensor& sinh_() const;
    Tensor& cosh_() const;
    Tensor& tanh_() const;
    Tensor& erf_() const;
    Tensor& erfc_() const;

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