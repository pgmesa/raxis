#pragma once

#include <raxis/autograd/graph.h>
#include <raxis/autograd/saved_variable.h>
#include <raxis/tensor/tensor.h>


namespace raxis::autograd {

class RAXIS_API AddBackward0 : public Node {
public:
    AddBackward0(const Tensor& a, const Tensor& b);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    bool a_requires_grad_;
    bool b_requires_grad_;
    DimVector a_shape_;
    DimVector b_shape_;
};


class RAXIS_API SubBackward0 : public Node {
public:
    SubBackward0(const Tensor& a, const Tensor& b);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    bool a_requires_grad_;
    bool b_requires_grad_;
    DimVector a_shape_;
    DimVector b_shape_;
};


class RAXIS_API MulBackward0 : public Node {
public:
    MulBackward0(const Tensor& a, const Tensor& b);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    bool a_requires_grad_;
    bool b_requires_grad_;
    DimVector a_shape_;
    DimVector b_shape_;
    SavedVariable a_{this, "self"};
    SavedVariable b_{this, "other"};
};


class RAXIS_API DivBackward0 : public Node {
public:
    DivBackward0(const Tensor& a, const Tensor& b);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    bool a_requires_grad_;
    bool b_requires_grad_;
    DimVector a_shape_;
    DimVector b_shape_;
    SavedVariable a_{this, "self"};
    SavedVariable b_{this, "other"};
};


// div with a rounding mode: the result is piecewise constant, so both
// gradients are zero
class RAXIS_API DivBackward2 : public Node {
public:
    DivBackward2(const Tensor& a, const Tensor& b);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    bool a_requires_grad_;
    bool b_requires_grad_;
    DimVector a_shape_;
    DimVector b_shape_;
};


// a - floor(a / b) * b: d/da = 1, d/db = -floor(a / b). The Scalar overload
// is RemainderBackward0 (only self), the Tensor one RemainderBackward1
class RAXIS_API RemainderBackward0 : public Node {
public:
    RemainderBackward0(const Tensor& a, const Tensor& b);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;
};

class RAXIS_API RemainderBackward1 : public Node {
public:
    RemainderBackward1(const Tensor& a, const Tensor& b);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    bool a_requires_grad_;
    bool b_requires_grad_;
    DimVector a_shape_;
    DimVector b_shape_;
    SavedVariable a_{this, "self"};
    SavedVariable b_{this, "other"};
};

// a - trunc(a / b) * b: as remainder, with trunc
class RAXIS_API FmodBackward0 : public Node {
public:
    FmodBackward0(const Tensor& a, const Tensor& b);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;
};

class RAXIS_API FmodBackward1 : public Node {
public:
    FmodBackward1(const Tensor& a, const Tensor& b);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    bool a_requires_grad_;
    bool b_requires_grad_;
    DimVector a_shape_;
    DimVector b_shape_;
    SavedVariable a_{this, "self"};
    SavedVariable b_{this, "other"};
};


class RAXIS_API PowBackward0 : public Node {
public:
    PowBackward0(const Tensor& base, Scalar exp);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    SavedVariable base_{this, "self"};
    Scalar exp_;
};

class RAXIS_API PowBackward1 : public Node {
public:
    PowBackward1(const Tensor& base, const Tensor& exp, const Tensor& result);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    bool base_requires_grad_;
    bool exp_requires_grad_;
    DimVector base_shape_;
    DimVector exp_shape_;
    SavedVariable base_{this, "self"};
    SavedVariable exp_{this, "exponent"};
    SavedVariable result_{this, "result"};
};

class RAXIS_API PowBackward2 : public Node {
public:
    PowBackward2(Scalar base, const Tensor& exp, const Tensor& result);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    Scalar base_;
    SavedVariable result_{this, "result"};
};


// MaximumBackward0: grad flows where a >= b
class RAXIS_API MaximumBackward0 : public Node {
public:
    MaximumBackward0(const Tensor& a, const Tensor& b);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;
private:
    bool a_requires_grad_;
    bool b_requires_grad_;
    DimVector a_shape_;
    DimVector b_shape_;
    SavedVariable a_{this, "self"};
    SavedVariable b_{this, "other"};
};

// MinimumBackward0: grad flows where a <= b
class RAXIS_API MinimumBackward0 : public Node {
public:
    MinimumBackward0(const Tensor& a, const Tensor& b);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;
private:
    bool a_requires_grad_;
    bool b_requires_grad_;
    DimVector a_shape_;
    DimVector b_shape_;
    SavedVariable a_{this, "self"};
    SavedVariable b_{this, "other"};
};

} // namespace raxis::autograd