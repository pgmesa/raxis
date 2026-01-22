
#include <raxis/core/array_iterator.hpp>

#include <raxis/array.hpp>
#include <raxis/core/shape.hpp>


namespace raxis {

    ArrayIteratorConfig& ArrayIteratorConfig::add_input(const NDArray& input) {
        inputs_.push_back(&input);
        num_inputs_++;
        return *this;
    }

    ArrayIteratorConfig& ArrayIteratorConfig::add_output(const NDArray& output) {
        outputs_.push_back(&output);
        num_outputs_++;
        return *this;
    }

    ArrayIterator ArrayIteratorConfig::build() {
        auto iter = ArrayIterator();
        iter.build(*this);
        return iter;
    }

    
    size_t ArrayIterator::ndim() const {
        return shape_.size();
    }

    size_t ArrayIterator::numel() const {
        return compute_numel(shape_);
    }

    int ArrayIterator::num_inputs() const {
        return num_operands() - num_inputs();
    }

    int ArrayIterator::num_outputs() const {
        return num_outputs_;
    }

    int ArrayIterator::num_operands() const {
        return static_cast<int>(operands_.size());
    }

    void ArrayIterator::build(const ArrayIteratorConfig& config) {


    }

}