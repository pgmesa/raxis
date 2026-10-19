from dataclasses import dataclass
from typing import Any

import raxis
from raxis import Tensor
from raxis.nn.modules import Module


@dataclass
class LayerInfo:
    """One call to a module during the summary's forward pass."""

    name: str  # path in the module tree, "" for the root
    type_name: str
    depth: int
    output_shape: Any = None  # a shape tuple, a list of them, or None
    num_params: int | None = None  # own parameters, None if counted in an earlier call


class ModelSummary:
    """Result of :func:`summary`: per-layer rows and parameter totals.

    ``str()`` (or ``print``) gives the table.
    """

    def __init__(self, layers: list[LayerInfo], params: list[Tensor]) -> None:
        self.layers = layers
        self.total_params = sum(p.numel() for p in params)
        self.trainable_params = sum(p.numel() for p in params if p.requires_grad)
        self.params_bytes = sum(p.numel() * p.dtype.itemsize for p in params)

    def __str__(self) -> str:
        rows = [("Layer (type)", "Output shape", "Param #")]
        for layer in self.layers:
            name = f"{layer.name} ({layer.type_name})" if layer.name else layer.type_name
            if layer.num_params is None:
                params = "(recursive)"
            else:
                params = f"{layer.num_params:,}" if layer.num_params else "--"
            rows.append(("  " * layer.depth + name, _format_shape(layer.output_shape), params))

        widths = [max(len(row[i]) for row in rows) for i in range(3)]
        line_width = sum(widths) + 4
        lines = []

        def add(row):
            lines.append(f"{row[0]:<{widths[0]}}  {row[1]:<{widths[1]}}  {row[2]:>{widths[2]}}")

        lines.append("=" * line_width)
        add(rows[0])
        lines.append("=" * line_width)
        for row in rows[1:]:
            add(row)
        lines.append("=" * line_width)
        lines.append(f"Total params: {self.total_params:,}")
        lines.append(f"Trainable params: {self.trainable_params:,}")
        lines.append(f"Non-trainable params: {self.total_params - self.trainable_params:,}")
        lines.append(f"Params size (MB): {self.params_bytes / 1e6:.2f}")
        lines.append("=" * line_width)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return str(self)


def _output_shape(output):
    if isinstance(output, Tensor):
        return tuple(output.shape)
    if isinstance(output, (tuple, list)):
        return [_output_shape(o) for o in output]
    return None


def _format_shape(shape) -> str:
    if shape is None:
        return "--"
    if isinstance(shape, list):
        return "[" + ", ".join(_format_shape(s) for s in shape) + "]"
    return str(shape)


def summary(model: Module, input: Tensor | tuple | list) -> ModelSummary:
    """Runs ``model`` on ``input`` and reports, per module call, its output
    shape and its own parameter count (as torchinfo does).

    ``input`` is a Tensor, or a tuple/list of positional arguments for
    ``forward``. The pass runs under ``no_grad`` and in the model's current
    mode (``train()``/``eval()``). Shared parameters count once in the totals;
    a module called again shows ``(recursive)``.
    """
    args = tuple(input) if isinstance(input, (tuple, list)) else (input,)
    layers: list[LayerInfo] = []
    # Open rows per module, filled with the output when its forward returns
    module_rows: dict[int, list[LayerInfo]] = {}
    seen: set[int] = set()
    hooked: set[int] = set()
    handles = []

    # A row is opened before forward, so parents come before their children
    for name, module in model.named_modules():
        # A module registered under two names is hooked once, with the first
        if id(module) in hooked:
            continue
        hooked.add(id(module))
        depth = name.count(".") + 1 if name else 0

        def pre_hook(module, args, name=name, depth=depth):
            num_params = sum(p.numel() for p in module._parameters.values() if p is not None)
            if num_params and id(module) in seen:
                num_params = None  # already counted in an earlier call
            seen.add(id(module))
            layers.append(LayerInfo(name, type(module).__name__, depth, num_params=num_params))
            module_rows.setdefault(id(module), []).append(layers[-1])

        def hook(module, args, output):
            module_rows[id(module)].pop().output_shape = _output_shape(output)

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(hook))

    try:
        with raxis.no_grad():
            model(*args)
    finally:
        for handle in handles:
            handle.remove()

    unique_params = list({p._cdata: p for p in model.parameters()}.values())
    return ModelSummary(layers, unique_params)
