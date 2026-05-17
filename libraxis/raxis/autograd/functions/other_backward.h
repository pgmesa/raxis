#pragma once

#include <raxis/autograd/graph.h>
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
};

class RAXIS_API BroadcastToBackward0 : public Node {
public:
    explicit BroadcastToBackward0(const Tensor& self);
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

} // namespace raxis::autograd