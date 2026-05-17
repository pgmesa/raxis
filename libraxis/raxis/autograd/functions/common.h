#pragma once

#include <raxis/autograd/variable.h>
#include <raxis/tensor/factories.h>


namespace raxis::autograd {

class NotImplementedBackward : public Node {
public:
    explicit NotImplementedBackward(std::string op_name)
        : op_name_(std::move(op_name)) {}

    std::string name() const override {
        return "NotImplementedBackward(" + op_name_ + ")";
    }

    TensorList apply(const TensorList&) override {
        RAXIS_CHECK(false,
            "the derivative for '", op_name_,
            "' is not implemented");
        return {};
    }

private:
    std::string op_name_;
};


class RAXIS_API ZeroBackward0 : public Node {
public:
    explicit ZeroBackward0(std::string op_name)
        : op_name_(std::move(op_name)) {}

    std::string name() const { return op_name_; }

    TensorList apply(const TensorList& grads) override  {
        return {zeros_like(grads[0])};
    }

private:
    std::string op_name_;
};

}