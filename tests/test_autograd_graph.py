import pytest
import raxis
import torch

from test_utils import check_match, make_grad_pair, randf


def walk(fn):
    """(name, [children]) of a grad_fn, comparable between raxis and torch."""
    if fn is None:
        return None
    # torch names AccumulateGrad "struct torch::autograd::AccumulateGrad" on MSVC
    return (fn.name().split("::")[-1], [walk(next_fn) for next_fn, _ in fn.next_functions])


def model(x, w, b):
    return ((x * w + b).sin() * x).sum()


class TestNodeInspection:

    def test_next_functions_match_torch(self):
        r = [make_grad_pair(randf(3, 4, seed=i)) for i in range(3)]
        out_r = model(*(p[0] for p in r))
        out_t = model(*(p[1] for p in r))
        assert walk(out_r.grad_fn) == walk(out_t.grad_fn)

    def test_input_without_grad_has_no_edge(self):
        x = raxis.rand((3,), requires_grad=True)
        y = x * raxis.rand((3,))
        (fn_x, nr_x), (fn_c, nr_c) = y.grad_fn.next_functions
        assert fn_x.name() == "AccumulateGrad" and nr_x == 0
        assert fn_c is None and nr_c == 0

    def test_one_accumulate_grad_per_leaf(self):
        x = raxis.rand((3,), requires_grad=True)
        y = x * x
        (a, _), (b, _) = y.grad_fn.next_functions
        assert a is b
        assert a.variable._cdata == x._cdata
        z = x.exp()
        assert z.grad_fn.next_functions[0][0] is a

    def test_output_nr_of_multi_output_op(self):
        x = raxis.rand((3, 2), requires_grad=True)
        y = x.unbind(0)[2] * 2.0
        assert y.grad_fn.next_functions[0][1] == 2

    def test_saved_tensors_match_torch(self):
        r, t = make_grad_pair(randf(3, 4))
        check_match(r.sin().grad_fn._saved_self, t.sin().grad_fn._saved_self)
        check_match(r.exp().grad_fn._saved_result, t.exp().grad_fn._saved_result)

    def test_saved_tensor_released_after_backward(self):
        r, t = make_grad_pair(randf(3, 4))
        for x in (r, t):
            y = x.sin()
            y.sum().backward()
            with pytest.raises(RuntimeError, match="backward through the graph a second time"):
                y.grad_fn._saved_self

    def test_saved_variables_listing(self):
        x = raxis.rand((3, 4), requires_grad=True)
        y = (x * 2.0).sin()
        [(name, tensor, released)] = y.grad_fn._saved_variables()
        assert (name, tuple(tensor.shape), released) == ("self", (3, 4), False)
        y.sum().backward()
        assert y.grad_fn._saved_variables() == [("self", None, True)]

    def test_unknown_attribute(self):
        x = raxis.rand((3,), requires_grad=True)
        with pytest.raises(AttributeError):
            x.sin().grad_fn._saved_other

    def test_sequence_nr_follows_creation(self):
        x = raxis.rand((3,), requires_grad=True)
        a = x * 2.0
        b = a.exp()
        assert b.grad_fn._sequence_nr() > a.grad_fn._sequence_nr()
