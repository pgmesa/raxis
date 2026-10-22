#pragma once

#include <raxis/autograd/graph.h>
#include <raxis/autograd/saved_variable.h>
#include <raxis/ops/loss.h>
#include <raxis/tensor/tensor.h>
#include <raxis/util/small_vector.h>


namespace raxis::autograd {

class RAXIS_API ReluBackward0 : public Node {
public:
    explicit ReluBackward0(const Tensor& result);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    SavedVariable result_{this, "result"};
};


class RAXIS_API SoftmaxBackward0 : public Node {
public:
    SoftmaxBackward0(const Tensor& result, int64_t dim);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    SavedVariable result_{this, "result"};
    int64_t dim_;
};


class RAXIS_API LogSoftmaxBackward0 : public Node {
public:
    LogSoftmaxBackward0(const Tensor& result, int64_t dim);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    SavedVariable result_{this, "result"};
    int64_t dim_;
};


class RAXIS_API NllLossBackward0 : public Node {
public:
    NllLossBackward0(
        const Tensor& self, const Tensor& target,
        Reduction reduction, int64_t ignore_index, double total_weight);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    SavedVariable self_{this, "self"};
    SavedVariable target_{this, "target"};
    Reduction reduction_;
    int64_t ignore_index_;
    double total_weight_;
};



// Saves input and weight unconditionally, as PyTorch: grad_weight needs the
// input and grad_input needs the weight, which is alive anyway as a parameter.
class RAXIS_API ConvolutionBackward0 : public Node {
public:
    ConvolutionBackward0(
        const Tensor& input, const Tensor& weight, const Tensor& bias,
        IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    SavedVariable input_{this, "input"};
    SavedVariable weight_{this, "weight"};
    DimVector stride_;
    DimVector padding_;
    DimVector dilation_;
    int64_t groups_;
    bool input_requires_grad_;
    bool weight_requires_grad_;
    bool bias_requires_grad_;
};

} // namespace raxis::autograd
