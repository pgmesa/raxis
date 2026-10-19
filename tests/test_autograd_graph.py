import shutil
import sys
import types

import pytest
import raxis
import torch

from raxis.autograd import graph
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


class TestGraphExport:

    def setup_method(self):
        self.w = raxis.rand((4, 2), requires_grad=True)
        self.b = raxis.rand((2,), requires_grad=True)
        self.x = raxis.rand((3, 4))
        self.out = ((self.x @ self.w) + self.b).relu().sum()
        self.params = {"w": self.w, "b": self.b}

    def test_to_dot_backward(self):
        dot = graph.to_dot(self.out, self.params)
        assert dot.startswith("digraph {")
        for name in ("SumBackward0", "ReluBackward0", "AddBackward0", "MmBackward0"):
            assert name in dot
        assert '"w\\n(4, 2) float32"' in dot
        assert '"b\\n(2,) float32"' in dot
        assert "result: (3, 2) float32" in dot
        # output -> sum -> relu -> add -> {mm, b}, mm -> w
        assert dot.count(" -> ") == 6

    def test_to_dot_forward(self):
        dot = graph.to_dot(self.out, self.params, view="forward")
        for op in ('"sum"', '"relu"', '"add"', '"mm"'):
            assert op in dot
        assert "Backward" not in dot
        assert dot.count(" -> ") == 6

    def test_forward_reverses_edges(self):
        def edges(text):
            return {tuple(line.strip().split(" -> ")) for line in text.splitlines() if " -> " in line}
        backward = edges(graph.to_dot(self.out))
        forward = edges(graph.to_dot(self.out, view="forward"))
        assert forward == {(dst, src) for src, dst in backward}

    def test_to_mermaid(self):
        text = graph.to_mermaid(self.out, self.params)
        assert text.startswith("flowchart TB")
        assert text.count(" --> ") == 6
        assert '["w<br/>(4, 2) float32"]:::leaf' in text
        assert 'out0["output<br/>() float32"]:::output' in text

    def test_hide_saved(self):
        assert "result:" not in graph.to_dot(self.out, show_saved=False)

    def test_released_saved_tensor(self):
        self.out.backward()
        assert "result: released" in graph.to_dot(self.out)

    def test_leaf_names_from_named_parameters(self):
        layer = raxis.nn.Linear(4, 2)
        dot = graph.to_dot(layer(self.x).sum(), dict(layer.named_parameters()))
        assert '"weight\\n(2, 4) float32"' in dot
        assert '"bias\\n(2,) float32"' in dot

    def test_several_outputs(self):
        a = self.w.sum()
        dot = graph.to_dot([a, a * 2.0])
        assert '"output 0\\n() float32"' in dot and '"output 1\\n() float32"' in dot
        assert dot.count("SumBackward0") == 1

    def test_leaf_output(self):
        assert '"leaf\\n(4, 2) float32"' in graph.to_dot(self.w)

    def test_deep_graph(self):
        y = self.w
        for _ in range(20_000):
            y = y + 1.0
        assert graph.to_mermaid(y).count("AddBackward0") == 20_000

    def test_bad_view(self):
        with pytest.raises(ValueError):
            graph.to_dot(self.out, view="sideways")


class TestRender:

    def setup_method(self):
        x = raxis.rand((3,), requires_grad=True)
        self.out = (x * 2.0).sum()

    def test_missing_python_package(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "graphviz", None)
        with pytest.raises(ImportError, match=r"raxis\[viz\]"):
            graph.render(self.out, tmp_path / "g.svg")

    def test_missing_binaries(self, monkeypatch, tmp_path):
        fake = types.ModuleType("graphviz")

        class ExecutableNotFound(RuntimeError):
            pass

        class Source:
            def __init__(self, source):
                pass

            def render(self, **kwargs):
                raise ExecutableNotFound("failed to execute 'dot'")

        fake.ExecutableNotFound = ExecutableNotFound
        fake.Source = Source
        monkeypatch.setitem(sys.modules, "graphviz", fake)
        with pytest.raises(RuntimeError, match="Graphviz binaries"):
            graph.render(self.out, tmp_path / "g.svg")

    def test_needs_extension(self, tmp_path):
        pytest.importorskip("graphviz")
        with pytest.raises(ValueError):
            graph.render(self.out, tmp_path / "g")

    @pytest.mark.skipif(shutil.which("dot") is None, reason="Graphviz binaries not installed")
    def test_render_svg(self, tmp_path):
        pytest.importorskip("graphviz")
        path = graph.render(self.out, tmp_path / "g.svg")
        assert path.read_text().lstrip().startswith(("<?xml", "<svg"))
