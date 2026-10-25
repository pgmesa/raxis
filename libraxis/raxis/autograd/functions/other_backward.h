#pragma once

#include <array>

#include <raxis/autograd/graph.h>
#include <raxis/autograd/saved_variable.h>
#include <raxis/tensor/tensor.h>


namespace raxis::autograd {

// ============================================================
// Cat / Stack / Unbind
// ============================================================

class RAXIS_API CatBackward0 : public Node {
public:
    CatBackward0(ArrayRef<Tensor> tensors, int64_t dim);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    int64_t dim_;
    std::vector<int64_t> sizes_;
    std::vector<bool> requires_grad_;
};

class RAXIS_API StackBackward0 : public Node {
public:
    StackBackward0(int64_t num_tensors, int64_t dim);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    int64_t num_tensors_;
    int64_t dim_;
};

class RAXIS_API UnbindBackward0 : public Node {
public:
    UnbindBackward0(const Tensor& self, int64_t dim, int64_t num_outputs);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    DimVector self_shape_;
    int64_t dim_;
};

class RAXIS_API SplitWithSizesBackward0 : public Node {
public:
    SplitWithSizesBackward0(const Tensor& self, IntArrayRef split_sizes, int64_t dim);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    DimVector self_shape_;
    DimVector split_sizes_;
    int64_t dim_;
};

class RAXIS_API WhereBackward0 : public Node {
public:
    WhereBackward0(const Tensor& condition, const Tensor& self, const Tensor& other);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    SavedVariable condition_{this, "condition"};
    DimVector self_shape_;
    DimVector other_shape_;
    bool self_requires_grad_;
    bool other_requires_grad_;
};

// clamp with tensor bounds (clamp.Tensor); min or max may be undefined
class RAXIS_API ClampBackward0 : public Node {
public:
    ClampBackward0(const Tensor& self, const Tensor& min, const Tensor& max);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    SavedVariable self_{this, "self"};
    SavedVariable min_{this, "min"};
    SavedVariable max_{this, "max"};
    std::array<bool, 3> needs_grad_;
    std::array<DimVector, 3> shapes_;
};

class RAXIS_API ClampBackward1 : public Node {
public:
    ClampBackward1(const Tensor& self, std::optional<Scalar> min, std::optional<Scalar> max);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    SavedVariable self_{this, "self"};
    std::optional<Scalar> min_;
    std::optional<Scalar> max_;
};

// ============================================================
// Gather / scatter / index_select
// ============================================================

class RAXIS_API GatherBackward0 : public Node {
public:
    GatherBackward0(const Tensor& self, int64_t dim, const Tensor& index);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    DimVector self_shape_;
    int64_t dim_;
    SavedVariable index_{this, "index"};
};

// scatter with a src tensor (inputs: self, src) or a value (input: self)
class RAXIS_API ScatterBackward0 : public Node {
public:
    ScatterBackward0(int64_t dim, const Tensor& index, bool has_src);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    int64_t dim_;
    SavedVariable index_{this, "index"};
    bool has_src_;
};

class RAXIS_API ScatterAddBackward0 : public Node {
public:
    ScatterAddBackward0(int64_t dim, const Tensor& index);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    int64_t dim_;
    SavedVariable index_{this, "index"};
};

class RAXIS_API IndexSelectBackward0 : public Node {
public:
    IndexSelectBackward0(const Tensor& self, int64_t dim, const Tensor& index);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    DimVector self_shape_;
    int64_t dim_;
    SavedVariable index_{this, "index"};
};

class RAXIS_API IndexAddBackward0 : public Node {
public:
    IndexAddBackward0(int64_t dim, const Tensor& index, const Tensor& source);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    int64_t dim_;
    SavedVariable index_{this, "index"};
    DimVector source_shape_;
};

// Advanced indexing x[indices]: the gradient goes back to the positions it
// was read from, adding up where an index repeats
class RAXIS_API IndexBackward0 : public Node {
public:
    IndexBackward0(const Tensor& self, ArrayRef<Tensor> indices);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    DimVector self_shape_;
    SavedVariableList indices_{this, "indices"};
};

class RAXIS_API IndexPutBackward0 : public Node {
public:
    IndexPutBackward0(const Tensor& self, ArrayRef<Tensor> indices, const Tensor& values, bool accumulate);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    SavedVariableList indices_{this, "indices"};
    DimVector values_shape_;
    bool accumulate_;
    std::array<bool, 2> needs_grad_;
};

class RAXIS_API MaskedFillBackward0 : public Node {
public:
    MaskedFillBackward0(const Tensor& self, const Tensor& mask);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    DimVector self_shape_;
    SavedVariable mask_{this, "mask"};
};

// ============================================================
// Shape ops
// ============================================================

class RAXIS_API ViewBackward0 : public Node {
public:
    explicit ViewBackward0(const Tensor& self);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    DimVector self_shape_;
};

class RAXIS_API TransposeBackward0 : public Node {
public:
    TransposeBackward0(int64_t dim0, int64_t dim1);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    int64_t dim0_;
    int64_t dim1_;
};

class RAXIS_API PermuteBackward0 : public Node {
public:
    explicit PermuteBackward0(IntArrayRef dims);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    DimVector inverse_dims_;
};

class RAXIS_API FlipBackward0 : public Node {
public:
    explicit FlipBackward0(IntArrayRef dims);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    DimVector dims_;
};

// tril and triu are linear: the gradient keeps the same triangle
class RAXIS_API TrilBackward0 : public Node {
public:
    explicit TrilBackward0(int64_t diagonal);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    int64_t diagonal_;
};

class RAXIS_API TriuBackward0 : public Node {
public:
    explicit TriuBackward0(int64_t diagonal);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    int64_t diagonal_;
};

class RAXIS_API UnsqueezeBackward0 : public Node {
public:
    explicit UnsqueezeBackward0(int64_t dim);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    int64_t dim_;
};

class RAXIS_API SqueezeBackward0 : public Node {
public:
    explicit SqueezeBackward0(const Tensor& self, std::optional<int64_t> dim);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    DimVector self_shape_;
    bool has_dim_;
};

class RAXIS_API ExpandBackward0 : public Node {
public:
    explicit ExpandBackward0(const Tensor& self);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    DimVector self_shape_;
};

class RAXIS_API SelectBackward0 : public Node {
public:
    SelectBackward0(const Tensor& self, int64_t dim, int64_t index);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    DimVector self_shape_;
    int64_t dim_;
    int64_t index_;
};

class RAXIS_API SliceBackward0 : public Node {
public:
    SliceBackward0(const Tensor& self, int64_t dim, int64_t start, int64_t stop, int64_t step);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    DimVector self_shape_;
    int64_t dim_;
    int64_t start_;
    int64_t stop_;
    int64_t step_;
};

// ============================================================
// Copies
// ============================================================

class RAXIS_API CloneBackward0 : public Node {
public:
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;
};

class RAXIS_API ToCopyBackward0 : public Node {
public:
    explicit ToCopyBackward0(const Tensor& self);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    Dtype self_dtype_;
};

} // namespace raxis::autograd