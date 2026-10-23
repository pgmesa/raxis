#pragma once

#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <raxis/core/dtype.h>
#include <raxis/util/exception.h>
#include <raxis/tensor/tensor.h>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/tensor_shape.h>
#include <raxis/tensor/shape_utils.h>
#include <raxis/autograd/graph.h>
#include <raxis/autograd/grad_mode.h>
#include <raxis/autograd/variable.h>
#include <raxis/autograd/autograd_meta.h>
#include <raxis/autograd/saved_variable.h>
#include <raxis/tensor/detail/tensor_impl.h>


namespace py = pybind11;


namespace raxis::python {

// From: torch/csrc/autograd/python_function.cpp
//
// A Python autograd.Function in the graph. Function.apply (raxis/autograd/
// function.py) runs forward under no_grad and hands the outputs to
// wrap_function_outputs, which creates a PyNode as their grad_fn. In the
// backward, PyNode::apply calls the Python backward(ctx, *grads).
//
// In PyTorch the ctx is the Python object of the node itself; here it is a
// separate Python object the node keeps alive.
class PyNode : public autograd::Node {
public:
    PyNode(py::object ctx, py::object backward, std::string name, std::vector<bool> is_tensor_input,
           std::vector<DimVector> input_shapes)
        : ctx_(std::move(ctx)), backward_(std::move(backward)), name_(std::move(name)),
          is_tensor_input_(std::move(is_tensor_input)), input_shapes_(std::move(input_shapes)) {}

    // The Python objects are released with the GIL held
    ~PyNode() override {
        py::gil_scoped_acquire gil;
        ctx_ = py::object();
        backward_ = py::object();
    }

    std::string name() const override { return name_; }

    // One per output of forward: zeros for a missing gradient when the ctx
    // asks to materialize them (undefined dtype for non-tensor outputs)
    void add_output_info(IntArrayRef shape, Dtype dtype) {
        output_shapes_.emplace_back(shape);
        output_dtypes_.push_back(dtype);
    }

    void save(const Tensor& tensor, bool is_output) { saved_.add(tensor, is_output); }

    autograd::TensorList apply(const autograd::TensorList& grads) override {
        py::gil_scoped_acquire gil;

        // From: torch/csrc/autograd/python_function.cpp (PyNode::to_py_args)
        const bool materialize = ctx_.attr("materialize_grads").cast<bool>();
        py::tuple args(grads.size() + 1);
        args[0] = ctx_;
        for (size_t i = 0; i < grads.size(); i++) {
            if (grads[i].defined()) {
                args[i + 1] = py::cast(grads[i]);
            } else if (materialize && output_dtypes_[i] != dtypes::Undefined) {
                args[i + 1] = py::cast(zeros(output_shapes_[i], TensorOptions().dtype(output_dtypes_[i])));
            } else {
                args[i + 1] = py::none();
            }
        }

        // The saved tensors are unpacked (version check) for this call only
        const autograd::TensorList unpacked = saved_.unpack();
        py::tuple saved(unpacked.size());
        for (size_t i = 0; i < unpacked.size(); i++) {
            saved[i] = unpacked[i].defined() ? py::cast(unpacked[i]) : py::none();
        }
        ctx_.attr("_saved_tensors") = saved;
        py::object result;
        try {
            result = backward_(*args);
        } catch (...) {
            ctx_.attr("_saved_tensors") = py::none();
            throw;
        }
        ctx_.attr("_saved_tensors") = py::none();
        return to_gradients(result);
    }

private:
    // From: torch/csrc/autograd/python_function.cpp (PyNode::apply, to_variable_list)
    autograd::TensorList to_gradients(const py::object& result) {
        py::tuple grads = py::isinstance<py::tuple>(result) ? py::tuple(result) : py::make_tuple(result);
        size_t num_grads = grads.size();
        const size_t num_inputs = is_tensor_input_.size();
        // Extra results are allowed if they are all None
        if (num_grads > num_inputs) {
            bool all_none = true;
            for (size_t i = num_inputs; i < num_grads; i++) {
                all_none &= grads[i].is_none();
            }
            if (all_none) num_grads = num_inputs;
        }
        RAXIS_CHECK(num_grads == num_inputs,
            "function ", name_, " returned an incorrect number of gradients (expected ",
            num_inputs, ", got ", num_grads, ")");

        autograd::TensorList result_grads;
        size_t tensor_idx = 0;
        for (size_t i = 0; i < num_inputs; i++) {
            py::handle grad = grads[i];
            if (!is_tensor_input_[i]) {
                RAXIS_CHECK(grad.is_none(),
                    "function ", name_, " returned a gradient different than None at position ",
                    i + 1, ", but the corresponding forward input was not a Variable");
                continue;
            }
            Tensor g;
            if (!grad.is_none()) {
                RAXIS_CHECK(py::isinstance<Tensor>(grad),
                    "expected Variable or None (got ", std::string(py::str(grad.get_type().attr("__name__"))), ")");
                g = grad.cast<Tensor>();
                g = to_input_shape(g, tensor_idx);
            }
            result_grads.push_back(std::move(g));
            tensor_idx++;
        }
        return result_grads;
    }

    // From: torch/csrc/autograd/engine.cpp (validate_outputs)
    // A gradient broadcast to more dims than its input is summed back down
    Tensor to_input_shape(const Tensor& grad, size_t tensor_idx) {
        const DimVector& shape = input_shapes_[tensor_idx];
        if (grad.shape() == IntArrayRef(shape)) {
            return grad;
        }
        bool expandable = true;
        try {
            expandable = IntArrayRef(broadcast_shapes(shape, grad.shape())) == grad.shape();
        } catch (const std::exception&) {
            expandable = false;
        }
        RAXIS_CHECK(expandable,
            "Function ", name_, " returned an invalid gradient at index ", tensor_idx,
            " - got ", grad.shape(), " but expected shape compatible with ", IntArrayRef(shape));
        return autograd::unbroadcast(grad, shape);
    }

    py::object ctx_;
    py::object backward_;
    std::string name_;
    std::vector<bool> is_tensor_input_;
    std::vector<DimVector> input_shapes_;
    std::vector<DimVector> output_shapes_;
    std::vector<Dtype> output_dtypes_;
    autograd::SavedVariableList saved_{this, "saved_tensors"};
};


inline std::unordered_set<detail::TensorImpl*> tensor_impls(const py::object& seq) {
    std::unordered_set<detail::TensorImpl*> impls;
    if (seq.is_none()) return impls;
    for (py::handle item : py::reinterpret_borrow<py::sequence>(seq)) {
        if (py::isinstance<Tensor>(item)) {
            impls.insert(item.cast<Tensor>().unsafe_get_tensor_impl());
        }
    }
    return impls;
}

// The output is the input itself: autograd needs a distinct tensor to give
// it this node as grad_fn, so it gets a view of the input
// (torch/csrc/autograd/custom_function.cpp, _view_as_self_with_no_grad)
inline Tensor view_as_self(const Tensor& t) {
    autograd::NoGradGuard guard;
    return make_view(t, t.storage_offset(), t.shape(), t.strides());
}

// From: torch/csrc/autograd/custom_function.cpp (_process_backward_mode_ad)
//       torch/csrc/autograd/python_function.cpp (process_outputs, _save_variables)
//
// Gives the outputs of forward their history: a PyNode when an input requires
// grad (is_executable) whose edges go to the tensor inputs. Handles what the
// ctx marked: dirty inputs (modified in place by forward: their history is
// rebased on the node), non-differentiable outputs, and the tensors saved
// for backward. Returns the outputs, with inputs returned as-is replaced by
// views of them.
inline py::object wrap_function_outputs(
    const py::object& ctx, const py::object& backward, const std::string& name,
    const py::tuple& inputs, const py::object& raw_outputs, bool is_executable
) {
    const bool is_tuple = py::isinstance<py::tuple>(raw_outputs);
    py::tuple outputs = is_tuple ? py::tuple(raw_outputs) : py::make_tuple(raw_outputs);
    const size_t num_outputs = outputs.size();

    std::vector<bool> is_tensor_input;
    std::vector<DimVector> input_shapes;
    std::vector<Tensor> tensor_inputs;
    for (py::handle input : inputs) {
        const bool is_tensor = py::isinstance<Tensor>(input);
        is_tensor_input.push_back(is_tensor);
        if (is_tensor) {
            tensor_inputs.push_back(input.cast<Tensor>());
            input_shapes.emplace_back(tensor_inputs.back().shape());
        }
    }
    std::unordered_set<detail::TensorImpl*> input_impls;
    for (const Tensor& t : tensor_inputs) {
        input_impls.insert(t.unsafe_get_tensor_impl());
    }
    // From: torch/csrc/autograd/python_function.cpp (_mark_dirty)
    // Forward may have written them without an in-place op that bumps the
    // version, so the version is bumped here
    const py::object dirty_tensors = py::getattr(ctx, "dirty_tensors", py::none());
    const auto dirty = tensor_impls(dirty_tensors);
    if (!dirty_tensors.is_none()) {
        for (py::handle item : py::reinterpret_borrow<py::sequence>(dirty_tensors)) {
            RAXIS_CHECK(py::isinstance<Tensor>(item), "mark_dirty can only accept variables");
            autograd::bump_version(item.cast<Tensor>());
        }
    }
    const auto non_differentiable = tensor_impls(py::getattr(ctx, "non_differentiable", py::none()));

    std::shared_ptr<PyNode> node;
    if (is_executable) {
        node = std::make_shared<PyNode>(ctx, backward, name, std::move(is_tensor_input), std::move(input_shapes));
        node->set_num_inputs(num_outputs);
        autograd::set_next_edges(node, tensor_inputs);
    }

    py::tuple results(num_outputs);
    std::unordered_set<detail::TensorImpl*> output_impls;
    int num_differentiable = 0;
    for (size_t i = 0; i < num_outputs; i++) {
        if (!py::isinstance<Tensor>(outputs[i])) {
            if (node) node->add_output_info({}, dtypes::Undefined);
            results[i] = outputs[i];
            continue;
        }
        Tensor var = outputs[i].cast<Tensor>();
        detail::TensorImpl* impl = var.unsafe_get_tensor_impl();
        const bool is_input = input_impls.count(impl) > 0;
        const bool is_modified = dirty.count(impl) > 0;
        const bool is_differentiable = node && non_differentiable.count(impl) == 0
            && var.dtype().is_floating();
        if (node) node->add_output_info(var.shape(), var.dtype());

        if (!is_differentiable) {
            if (!var.requires_grad()) {
                if (is_input && !is_modified) var = view_as_self(var);
            } else if (is_input) {
                // A detached alias, instead of changing the input's requires_grad
                var = var.detach();
            }
        } else if (is_modified) {
            RAXIS_CHECK(!(var.is_leaf() && var.requires_grad()),
                "a leaf Variable that requires grad has been used in an in-place operation.");
            RAXIS_CHECK(is_input,
                "ctx.mark_dirty() received a tensor that was not an input. Only input Tensors "
                "that have been mutated should be passed to ctx.mark_dirty().");
            RAXIS_CHECK(!(var.is_view() && num_outputs > 1),
                "If your Function modifies inplace an input that is a view of another Tensor, your "
                "Function cannot return more than one Tensor. This is not supported by the current "
                "autograd engine. You should either make sure the input is not a view (using "
                ".clone() for example) or make your Function only return one Tensor (potentially "
                "splitting it into two Functions: one doing the inplace that returns a single Tensor "
                "and a second one that does the other operations).");
            autograd::check_inplace(var, true);
            autograd::rebase_history(var, node, static_cast<uint32_t>(i));
        } else {
            if (is_input) var = view_as_self(var);
            autograd::set_autograd_meta(var, node, static_cast<uint32_t>(i));
        }

        // A view returned by the Function cannot be modified in place: the
        // CopySlices of that in-place op would bypass its custom backward
        if (!(is_input && is_modified) && var.is_view()) {
            var.unsafe_get_tensor_impl()->set_creation_meta(autograd::CreationMeta::InCustomFunction);
        }
        num_differentiable += is_differentiable;
        output_impls.insert(var.unsafe_get_tensor_impl());
        output_impls.insert(impl);
        results[i] = py::cast(var);
    }

    // Several differentiable outputs: their views cannot be modified in place
    if (num_differentiable > 1) {
        for (py::handle result : results) {
            if (py::isinstance<Tensor>(result)) {
                Tensor t = result.cast<Tensor>();
                if (t.is_view()) {
                    t.unsafe_get_tensor_impl()->set_creation_meta(autograd::CreationMeta::MultiOutputNode);
                }
            }
        }
    }
    for (detail::TensorImpl* impl : dirty) {
        RAXIS_CHECK(output_impls.count(impl) > 0,
            "Some elements marked as dirty during the forward method were not returned as output. "
            "The inputs that are modified inplace must all be outputs of the Function.");
    }

    // From: torch/csrc/autograd/python_function.cpp (_save_variables)
    // An output is saved without its grad_fn (it would own the node)
    py::object to_save = py::getattr(ctx, "to_save", py::none());
    if (node && !to_save.is_none()) {
        for (py::handle item : py::reinterpret_borrow<py::sequence>(to_save)) {
            if (item.is_none()) {
                node->save(Tensor(), false);
                continue;
            }
            RAXIS_CHECK(py::isinstance<Tensor>(item),
                "save_for_backward can only save variables, but argument is of type ",
                std::string(py::str(item.get_type().attr("__name__"))));
            Tensor t = item.cast<Tensor>();
            node->save(t, t.grad_fn() == node);
        }
    }
    // The ctx is kept alive by the node: holding the tensors here too would
    // close a cycle through the outputs
    for (const char* attr : {"to_save", "dirty_tensors", "non_differentiable"}) {
        if (py::hasattr(ctx, attr)) py::delattr(ctx, attr);
    }

    return is_tuple ? py::object(results) : py::object(results[0]);
}

// From: torch/csrc/autograd/engine.cpp (Engine::execute)
// backward(create_graph=True) leaves .grad of each leaf with a graph that
// ends in the leaf's AccumulateGrad, which holds the leaf: a cycle that is
// only broken by setting .grad to None
inline void warn_backward_create_graph() {
    if (PyErr_WarnEx(PyExc_UserWarning,
            "Using backward() with create_graph=True will create a reference cycle between the "
            "parameter and its gradient which can cause a memory leak. We recommend using "
            "autograd.grad when creating the graph to avoid this. If you have to use this "
            "function, make sure to reset the .grad fields of your parameters to None after use "
            "to break the cycle and avoid the leak.", 1) < 0) {
        throw py::error_already_set();
    }
}

} // namespace raxis::python
