"""Inspect and export the autograd graph of a tensor.

What raxis records during the forward pass is the backward graph: every op
whose inputs require grad creates a node (``MulBackward0``, ...) whose
``next_functions`` point to the nodes that produced its inputs, down to one
``AccumulateGrad`` per leaf. ``backward()`` walks it from the output to the
leaves.

Read with the edges reversed, the same graph is the forward graph: leaves
flow into ops, and the last op produces the output. ``view="forward"`` draws
it that way, labelling each node with its forward op (``MulBackward0`` ->
``mul``). Inputs that do not require grad (constants, data) are not recorded,
so they do not appear in either view.

``to_dot`` and ``to_mermaid`` return text and need no dependencies (paste the
Mermaid text in a Markdown file or https://mermaid.live). ``render`` writes an
image through the optional ``graphviz`` package (``pip install raxis[viz]``),
which also needs the Graphviz binaries installed on the system.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import raxis


__all__ = ["to_dot", "to_mermaid", "render"]


_FILL = {
    "op": "lightgrey",
    "leaf": "lightblue",
    "output": "darkolivegreen1",
}

_MERMAID_FILL = {
    "op": "#d3d3d3",
    "leaf": "#add8e6",
    "output": "#caff70",
}


@dataclass
class _Graph:
    # node id -> (label lines, kind)
    nodes: dict = field(default_factory=dict)
    edges: list = field(default_factory=list)


def _describe(tensor):
    return f"{tuple(tensor.shape)} {tensor.dtype}"


def _forward_name(node_name):
    # MulBackward0 -> mul, LogSoftmaxBackward0 -> log_softmax, Log1PBackward0 -> log1p
    op = re.sub(r"Backward\d*$", "", node_name)
    return re.sub(r"(?<=[a-z])(?=[A-Z])", "_", op).lower()


def _as_tensors(outputs):
    if isinstance(outputs, raxis.Tensor):
        return [outputs]
    if isinstance(outputs, Sequence):
        return list(outputs)
    raise TypeError(f"expected a Tensor or a sequence of Tensors, got {type(outputs).__name__}")


def _build(outputs, params, view, show_saved):
    if view not in ("backward", "forward"):
        raise ValueError(f"view must be 'backward' or 'forward', got {view!r}")
    names = {}
    if params is not None:
        if not isinstance(params, Mapping):
            params = dict(params)
        names = {t._cdata: name for name, t in params.items()}

    graph = _Graph()
    ids = {}  # Python wrapper -> node id; pybind returns the same wrapper while it is alive

    def node_id(fn):
        if fn not in ids:
            ids[fn] = f"n{len(ids)}"
        return ids[fn]

    def edge(src, dst):
        # Stored in gradient direction (backward view); forward reverses it
        graph.edges.append((src, dst) if view == "backward" else (dst, src))

    def leaf_label(tensor):
        name = names.get(tensor._cdata)
        return [name if name is not None else "leaf", _describe(tensor)]

    def op_label(fn):
        if view == "forward":
            return [_forward_name(fn.name())]
        lines = [fn.name()]
        if show_saved:
            for name, tensor, released in fn._saved_variables():
                if released:
                    lines.append(f"{name}: released")
                else:
                    param = names.get(tensor._cdata)
                    lines.append(f"{name}: {param + ' ' if param else ''}{_describe(tensor)}")
        return lines

    # Iterative DFS: the graph can be deeper than the Python stack
    stack = []
    outputs = _as_tensors(outputs)
    for i, out in enumerate(outputs):
        out_id = f"out{i}"
        label = "output" if len(outputs) == 1 else f"output {i}"
        if out.grad_fn is None:
            # A leaf: the graph is just the tensor
            graph.nodes[out_id] = (leaf_label(out) if out.requires_grad else [label, _describe(out)],
                                   "leaf" if out.requires_grad else "output")
            continue
        graph.nodes[out_id] = ([label, _describe(out)], "output")
        edge(out_id, node_id(out.grad_fn))
        stack.append(out.grad_fn)

    while stack:
        fn = stack.pop()
        fid = node_id(fn)
        if fid in graph.nodes:
            continue
        if isinstance(fn, raxis.AccumulateGrad):
            graph.nodes[fid] = (leaf_label(fn.variable), "leaf")
            continue
        graph.nodes[fid] = (op_label(fn), "op")
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                edge(fid, node_id(next_fn))
                stack.append(next_fn)
    return graph


def to_dot(outputs, params=None, *, view="backward", show_saved=True):
    """Graphviz DOT source of the autograd graph of ``outputs``.

    Args:
        outputs: a Tensor or a sequence of Tensors.
        params: optional ``{name: tensor}`` (e.g. ``dict(model.named_parameters())``)
            to label leaves and saved tensors with their names.
        view: ``"backward"`` (gradient flow, autograd nodes) or ``"forward"``
            (data flow, forward ops).
        show_saved: list the tensors each node saved for backward (backward view).
    """
    graph = _build(outputs, params, view, show_saved)

    def quote(lines):
        text = "\\n".join(line.replace("\\", "\\\\").replace('"', '\\"') for line in lines)
        return f'"{text}"'

    out = [
        "digraph {",
        "    graph [rankdir=TB]",
        '    node [shape=box, style="rounded,filled", fontname="monospace", fontsize=10]',
    ]
    for nid, (lines, kind) in graph.nodes.items():
        shape = ", shape=ellipse" if kind == "op" and view == "forward" else ""
        out.append(f"    {nid} [label={quote(lines)}, fillcolor={_FILL[kind]}{shape}]")
    for src, dst in graph.edges:
        out.append(f"    {src} -> {dst}")
    out.append("}")
    return "\n".join(out) + "\n"


def to_mermaid(outputs, params=None, *, view="backward", show_saved=True):
    """Mermaid flowchart of the autograd graph of ``outputs``.

    Same arguments as :func:`to_dot`.
    """
    graph = _build(outputs, params, view, show_saved)

    def quote(lines):
        return '"' + "<br/>".join(line.replace('"', "#quot;") for line in lines) + '"'

    out = ["flowchart TB"]
    for nid, (lines, kind) in graph.nodes.items():
        box = f"({quote(lines)})" if kind == "op" else f"[{quote(lines)}]"
        out.append(f"    {nid}{box}:::{kind}")
    for src, dst in graph.edges:
        out.append(f"    {src} --> {dst}")
    for kind, fill in _MERMAID_FILL.items():
        out.append(f"    classDef {kind} fill:{fill},stroke:#555,color:#000")
    return "\n".join(out) + "\n"


def render(outputs, path, params=None, *, view="backward", show_saved=True):
    """Render the graph to an image file through Graphviz.

    The format comes from the extension of ``path`` (``.svg``, ``.png``,
    ``.pdf``, ...). Needs the ``graphviz`` Python package
    (``pip install raxis[viz]``) and the Graphviz binaries (the ``dot``
    command) on the PATH. Returns the path of the written file.
    """
    try:
        import graphviz
    except ImportError:
        raise ImportError(
            "render() needs the graphviz Python package: pip install 'raxis[viz]'. "
            "to_dot() and to_mermaid() work without it."
        ) from None

    path = Path(path)
    if not path.suffix:
        raise ValueError(f"{path} has no extension: use .svg, .png, .pdf, ...")
    source = graphviz.Source(to_dot(outputs, params, view=view, show_saved=show_saved))
    try:
        source.render(outfile=path, cleanup=True)
    except graphviz.ExecutableNotFound:
        raise RuntimeError(
            "render() needs the Graphviz binaries (the `dot` command) on the PATH; "
            "the graphviz Python package does not include them. Install Graphviz "
            "(https://graphviz.org/download: winget install graphviz, "
            "apt install graphviz, brew install graphviz) or use to_dot() / to_mermaid()."
        ) from None
    return path
