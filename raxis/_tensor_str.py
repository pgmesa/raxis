
import contextlib
import dataclasses
import math
from typing import Any, Optional


@dataclasses.dataclass
class _PrinterOptions:
    precision: int = 4
    threshold: float = 1000
    edgeitems: int = 3
    linewidth: int = 80
    sci_mode: Optional[bool] = None


PRINT_OPTS = _PrinterOptions()


def set_printoptions(
    precision=None,
    threshold=None,
    edgeitems=None,
    linewidth=None,
    profile=None,
    sci_mode=None,
):
    """Set options for printing.

    Args:
        precision: Number of digits of precision for floating point output
            (default = 4).
        threshold: Total number of array elements which trigger summarization
            rather than full repr (default = 1000).
        edgeitems: Number of array items in summary at beginning and end of
            each dimension (default = 3).
        linewidth: The number of characters per line for the purpose of
            inserting line breaks (default = 80). Thresholded matrices will
            ignore this parameter.
        profile: Sane defaults for pretty printing. Can override with any of
            the above options. (any one of 'default', 'short', 'full')
        sci_mode: Enable (True) or disable (False) scientific notation. If
            None (default) is specified, the value is automatically chosen by
            the formatter.
    """
    if profile is not None:
        if profile == "default":
            PRINT_OPTS.precision = 4
            PRINT_OPTS.threshold = 1000
            PRINT_OPTS.edgeitems = 3
            PRINT_OPTS.linewidth = 80
        elif profile == "short":
            PRINT_OPTS.precision = 2
            PRINT_OPTS.threshold = 1000
            PRINT_OPTS.edgeitems = 2
            PRINT_OPTS.linewidth = 80
        elif profile == "full":
            PRINT_OPTS.precision = 4
            PRINT_OPTS.threshold = float("inf")
            PRINT_OPTS.edgeitems = 3
            PRINT_OPTS.linewidth = 80

    if precision is not None:
        PRINT_OPTS.precision = precision
    if threshold is not None:
        PRINT_OPTS.threshold = threshold
    if edgeitems is not None:
        PRINT_OPTS.edgeitems = edgeitems
    if linewidth is not None:
        PRINT_OPTS.linewidth = linewidth
    PRINT_OPTS.sci_mode = sci_mode


def get_printoptions() -> dict[str, Any]:
    """Gets the current options for printing."""
    return dataclasses.asdict(PRINT_OPTS)


@contextlib.contextmanager
def printoptions(**kwargs):
    """Context manager that temporarily changes the print options."""
    old_kwargs = get_printoptions()
    set_printoptions(**kwargs)
    try:
        yield
    finally:
        set_printoptions(**old_kwargs)


# ============================================================================
# Dtype helpers
# ============================================================================
_DEFAULT_DTYPE_NAMES = frozenset({"float32", "int64", "bool"})


# ============================================================================
# Formatter: decides how to print each scalar value
# ============================================================================

class _Formatter:
    """Scans (possibly summarized) tensor values to choose a format string.

    Determines:
        - floating_dtype: whether the tensor holds floats
        - int_mode: whether all float values happen to be integers (e.g. 1.0)
        - sci_mode: whether to use scientific notation
        - max_width: column width for right-aligned printing
    """

    def __init__(self, values: list):
        """
        Args:
            values: a flat Python list of all (or summarized) scalar values in
                    the tensor, obtained via tensor.tolist() or
                    _get_summarized_data().
        """
        self.floating_dtype = isinstance(values[0], float) if values else False
        self.int_mode = True
        self.sci_mode = False
        self.max_width = 1

        if not self.floating_dtype:
            for v in values:
                self.max_width = max(self.max_width, len(f"{v}"))
            return

        # --- Floating path ---
        nonzero_finite = [
            v for v in values
            if v != 0.0 and math.isfinite(v)
        ]

        if not nonzero_finite:
            return

        abs_vals = [abs(v) for v in nonzero_finite]
        abs_min = min(abs_vals)
        abs_max = max(abs_vals)

        # Check if all values are integer-valued floats
        for v in nonzero_finite:
            if v != math.floor(v):
                self.int_mode = False
                break

        # Decide scientific notation
        if PRINT_OPTS.sci_mode is not None:
            self.sci_mode = PRINT_OPTS.sci_mode
        else:
            self.sci_mode = (
                abs_max / abs_min > 1000.0
                or abs_max > 1.0e8
                or abs_min < 1.0e-4
            )

        # Compute max_width
        if self.int_mode:
            if self.sci_mode:
                for v in nonzero_finite:
                    s = f"{{:.{PRINT_OPTS.precision}e}}".format(v)
                    self.max_width = max(self.max_width, len(s))
            else:
                for v in nonzero_finite:
                    s = f"{v:.0f}"
                    self.max_width = max(self.max_width, len(s) + 1)
        else:
            if self.sci_mode:
                for v in nonzero_finite:
                    s = f"{{:.{PRINT_OPTS.precision}e}}".format(v)
                    self.max_width = max(self.max_width, len(s))
            else:
                for v in nonzero_finite:
                    s = f"{{:.{PRINT_OPTS.precision}f}}".format(v)
                    self.max_width = max(self.max_width, len(s))

    @property
    def width(self) -> int:
        return self.max_width

    def format(self, value) -> str:
        if self.floating_dtype:
            if self.sci_mode:
                ret = f"{{:{self.max_width}.{PRINT_OPTS.precision}e}}".format(value)
            elif self.int_mode:
                ret = f"{value:.0f}"
                if not (math.isinf(value) or math.isnan(value)):
                    ret += "."
            else:
                ret = f"{{:.{PRINT_OPTS.precision}f}}".format(value)
        else:
            ret = f"{value}"
        return " " * (self.max_width - len(ret)) + ret


# ============================================================================
# Summarization: extract edge items to avoid scanning all values
# ============================================================================

def _get_summarized_data(tensor) -> list:
    """Return a flat Python list of the edge-item values for the Formatter.

    For large tensors, instead of calling tensor.tolist() (which would
    materialize everything), we only grab the first and last `edgeitems`
    along each dimension recursively, then flatten.
    """
    if tensor.numel() <= 2 * PRINT_OPTS.edgeitems:
        return _flatten(tensor.tolist())

    return _summarize_recursive(tensor)


def _summarize_recursive(tensor) -> list:
    dim = tensor.dim()
    if dim == 0:
        return [tensor.item()]
    if dim == 1:
        n = tensor.shape[0]
        e = PRINT_OPTS.edgeitems
        if n > 2 * e:
            head = tensor[:e].tolist()
            tail = tensor[-e:].tolist()
            return head + tail
        return tensor.tolist()

    # nD: recurse on first and last edgeitems along dim 0
    n = tensor.shape[0]
    e = PRINT_OPTS.edgeitems
    if n > 2 * e:
        parts = [tensor[i] for i in range(e)]
        parts += [tensor[i] for i in range(n - e, n)]
    else:
        parts = [tensor[i] for i in range(n)]
    result = []
    for sub in parts:
        result.extend(_summarize_recursive(sub))
    return result


def _flatten(nested) -> list:
    """Flatten a nested Python list into a 1D list."""
    if not isinstance(nested, list):
        return [nested]
    result = []
    for item in nested:
        result.extend(_flatten(item))
    return result


# ============================================================================
# Core recursive string building
# ============================================================================

def _scalar_str(tensor, formatter) -> str:
    return formatter.format(tensor.item())


def _vector_str(tensor, indent, summarize, formatter) -> str:
    element_length = formatter.width + 2  # +2 for ", "
    elements_per_line = max(
        1, int(math.floor((PRINT_OPTS.linewidth - indent) / element_length))
    )

    if summarize and not PRINT_OPTS.edgeitems:
        data = ["..."]
    elif summarize and tensor.shape[0] > 2 * PRINT_OPTS.edgeitems:
        e = PRINT_OPTS.edgeitems
        data = (
            [formatter.format(v) for v in tensor[:e].tolist()]
            + [" ..."]
            + [formatter.format(v) for v in tensor[-e:].tolist()]
        )
    else:
        data = [formatter.format(v) for v in tensor.tolist()]

    data_lines = [
        data[i : i + elements_per_line]
        for i in range(0, len(data), elements_per_line)
    ]
    lines = [", ".join(line) for line in data_lines]
    return "[" + ("," + "\n" + " " * (indent + 1)).join(lines) + "]"


def _tensor_str_with_formatter(tensor, indent, summarize, formatter) -> str:
    dim = tensor.dim()

    if dim == 0:
        return _scalar_str(tensor, formatter)

    if dim == 1:
        return _vector_str(tensor, indent, summarize, formatter)

    n = tensor.shape[0]
    e = PRINT_OPTS.edgeitems

    if summarize and n > 2 * e:
        slices = (
            [
                _tensor_str_with_formatter(tensor[i], indent + 1, summarize, formatter)
                for i in range(e)
            ]
            + ["..."]
            + [
                _tensor_str_with_formatter(tensor[i], indent + 1, summarize, formatter)
                for i in range(n - e, n)
            ]
        )
    else:
        slices = [
            _tensor_str_with_formatter(tensor[i], indent + 1, summarize, formatter)
            for i in range(n)
        ]

    separator = "," + "\n" * (dim - 1) + " " * (indent + 1)
    return "[" + separator.join(slices) + "]"


def _tensor_str(tensor, indent) -> str:
    if tensor.numel() == 0:
        return "[]"

    summarize = tensor.numel() > PRINT_OPTS.threshold

    if summarize:
        flat_values = _get_summarized_data(tensor)
    else:
        flat_values = _flatten(tensor.tolist())

    formatter = _Formatter(flat_values)
    return _tensor_str_with_formatter(tensor, indent, summarize, formatter)


# ============================================================================
# Suffix building (dtype, device)
# ============================================================================

def _add_suffixes(tensor_str, suffixes, indent) -> str:
    parts = [tensor_str]
    last_line_len = len(tensor_str) - tensor_str.rfind("\n") + 1

    for suffix in suffixes:
        suffix_len = len(suffix)
        if last_line_len + suffix_len + 2 > PRINT_OPTS.linewidth:
            parts.append(",\n" + " " * indent + suffix)
            last_line_len = indent + suffix_len
        else:
            parts.append(", " + suffix)
            last_line_len += suffix_len + 2

    parts.append(")")
    return "".join(parts)


# ============================================================================
# Public entry point
# ============================================================================

def _str(tensor) -> str:
    """Main entry point for tensor.__repr__().

    Produces output like:
        raxis([1., 2., 3.])
        raxis([[1, 2],
               [3, 4]], dtype=int32)
    """
    prefix = "raxis("
    indent = len(prefix)
    suffixes = []

    # Show device if not the default (cpu)
    if tensor.device.type != "cpu":
        suffixes.append(f"device='{tensor.device}'")

    # Show dtype if not one of the "default" types
    if tensor.dtype.name not in _DEFAULT_DTYPE_NAMES:
        suffixes.append(f"dtype={tensor.dtype.name}")
        
    if tensor.requires_grad:
        if tensor.grad_fn is not None:
            suffixes.append(f"grad_fn=<{tensor.grad_fn.name()}>")
        else:
            suffixes.append("requires_grad=True")

    # Empty tensor
    if tensor.numel() == 0:
        tensor_str = "[]"
        # Show shape explicitly if not 1D (matches PyTorch behavior)
        if tensor.dim() != 1:
            suffixes.insert(0, f"shape={tuple(tensor.shape)}")
    else:
        # The values of a CUDA tensor are formatted from one copy on the CPU
        data = tensor if tensor.device.type == "cpu" else tensor.detach().cpu()
        tensor_str = _tensor_str(data, indent)

    return _add_suffixes(prefix + tensor_str, suffixes, indent)