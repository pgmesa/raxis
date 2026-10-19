#pragma once

#include <raxis/autograd/graph.h>
#include <raxis/autograd/saved_variable.h>
#include <raxis/tensor/tensor.h>


namespace raxis::autograd {

class RAXIS_API DotBackward0 : public Node {
public:
    DotBackward0(const Tensor& self, const Tensor& other);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    bool self_requires_grad_;
    bool other_requires_grad_;
    SavedVariable self_{this, "self"};
    SavedVariable other_{this, "other"};
};

class RAXIS_API MvBackward0 : public Node {
public:
    MvBackward0(const Tensor& mat, const Tensor& vec);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    bool mat_requires_grad_;
    bool vec_requires_grad_;
    SavedVariable mat_{this, "mat"};
    SavedVariable vec_{this, "vec"};
};

class RAXIS_API MmBackward0 : public Node {
public:
    MmBackward0(const Tensor& mat1, const Tensor& mat2);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    bool mat1_requires_grad_;
    bool mat2_requires_grad_;
    SavedVariable mat1_{this, "mat1"};
    SavedVariable mat2_{this, "mat2"};
};

class RAXIS_API BmmBackward0 : public Node {
public:
    BmmBackward0(const Tensor& self, const Tensor& mat2);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    bool self_requires_grad_;
    bool mat2_requires_grad_;
    SavedVariable self_{this, "self"};
    SavedVariable mat2_{this, "mat2"};
};

class RAXIS_API AddmmBackward0 : public Node {
public:
    AddmmBackward0(const Tensor& self, const Tensor& mat1, const Tensor& mat2,
                   Scalar beta, Scalar alpha);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    bool self_requires_grad_;
    bool mat1_requires_grad_;
    bool mat2_requires_grad_;
    DimVector self_shape_;
    SavedVariable mat1_{this, "mat1"};
    SavedVariable mat2_{this, "mat2"};
    Scalar beta_;
    Scalar alpha_;
};

class RAXIS_API AddmvBackward0 : public Node {
public:
    AddmvBackward0(const Tensor& self, const Tensor& mat, const Tensor& vec,
                   Scalar beta, Scalar alpha);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    bool self_requires_grad_;
    bool mat_requires_grad_;
    bool vec_requires_grad_;
    DimVector self_shape_;
    SavedVariable mat_{this, "mat"};
    SavedVariable vec_{this, "vec"};
    Scalar beta_;
    Scalar alpha_;
};

class RAXIS_API AddbmmBackward0 : public Node {
public:
    AddbmmBackward0(const Tensor& self, const Tensor& batch1, const Tensor& batch2,
                    Scalar beta, Scalar alpha);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    bool self_requires_grad_;
    bool batch1_requires_grad_;
    bool batch2_requires_grad_;
    DimVector self_shape_;
    SavedVariable batch1_{this, "batch1"};
    SavedVariable batch2_{this, "batch2"};
    Scalar beta_;
    Scalar alpha_;
    int64_t batch_size_;
};

}