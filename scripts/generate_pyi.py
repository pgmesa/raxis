import re
import sys
import shutil
from pathlib import Path

project_path = Path(__file__).parent.parent.absolute()
if project_path not in sys.path:
    sys.path.append(str(project_path))

import pybind11_stubgen


cpp_module_name = '_C'

REMOVE_LINES: list[str] = [
]

EXTRA_IMPORTS: list[str] = [
    'import builtins',
]

TYPE_ALIASES: str = """
_int = builtins.int
_float = builtins.float
_bool = builtins.bool

Number = _int | _float | _bool
Size = list[_int] | tuple[_int, ...]
"""

# Simple string replacements applied in order.
# Use \b in the pattern to opt into regex word-boundary matching.
TEXT_REPLACEMENTS: list[tuple[str, str]] = [
    ('numpy.dtype[typing.Any]',                   'dtype'),
    (r'\blist\[int\] \| tuple\[int, ...\]',       'Size'),
    (r'\bint \| float \| bool\b',                 'Number'),
    ('-> bool:',                                  '-> _bool:'),
    (': bool',                                    ': _bool')
]


# raxis/__init__.py replaces some Tensor methods with Python ones after
# loading _C. Their stubs are written here, as torch writes them in its .pyi.
TENSOR_METHODS: dict[str, str] = {
    '__repr__': 'def __repr__(self) -> str: ...',
    '__hash__': 'def __hash__(self) -> _int: ...',
    '__float__': 'def __float__(self) -> _float: ...',
    '__int__': 'def __int__(self) -> _int: ...',
    'item': 'def item(self) -> Number: ...',
    'size': '''@typing.overload
def size(self, dim: None = None) -> tuple[_int, ...]: ...
@typing.overload
def size(self, dim: _int) -> _int: ...''',
    'stride': '''@typing.overload
def stride(self, dim: None = None) -> tuple[_int, ...]: ...
@typing.overload
def stride(self, dim: _int) -> _int: ...''',
    'nonzero': '''@typing.overload
def nonzero(self, *, as_tuple: typing.Literal[False] = False) -> Tensor: ...
@typing.overload
def nonzero(self, *, as_tuple: typing.Literal[True]) -> tuple[Tensor, ...]: ...''',
    'shape': '@property\ndef shape(self) -> tuple[_int, ...]: ...',
    'strides': '@property\ndef strides(self) -> tuple[_int, ...]: ...',
    '_base': '@property\ndef _base(self) -> Tensor | None: ...',
}

# Shapes also taken as separate ints (x.view(3, 4), raxis.zeros(3, 4)), see
# _sizes_as_varargs in raxis/__init__.py
TENSOR_SIZE_VARARGS = ['view', 'reshape', 'permute', 'expand', 'flip', 'repeat',
                       'new_empty', 'new_zeros', 'new_ones']
FUNCTION_SIZE_VARARGS = ['empty', 'zeros', 'ones', 'rand', 'randn']


def find_blocks(text: str, name: str, indent: str) -> list[re.Match]:
    """Every definition of `name` at this indentation (overloads included):
    its decorators, the def line and the indented body."""
    pattern = (rf'(?:^{indent}@.*\n)*^{indent}def {re.escape(name)}\(.*\n'
               rf'(?:(?:{indent}    .*|[ \t]+)\n)*')
    return list(re.finditer(pattern, text, flags=re.MULTILINE))


def replace_definition(text: str, name: str, stub: str, indent: str) -> str:
    blocks = find_blocks(text, name, indent)
    if not blocks:
        raise RuntimeError(f'generate_pyi: no stub for {name} to replace')
    new = ''.join(f'{indent}{line}\n' for line in stub.splitlines())
    start, end = blocks[0].start(), blocks[-1].end()
    return text[:start] + new + text[end:]


def add_size_varargs(text: str, name: str, indent: str) -> str:
    """def f(self, shape: Size, ...) -> R  becomes two overloads, the second
    with def f(self, *shape: _int, ...) -> R."""
    blocks = find_blocks(text, name, indent)
    if len(blocks) != 1:
        raise RuntimeError(f'generate_pyi: expected one stub for {name}, found {len(blocks)}')
    def_line = next(l for l in blocks[0].group(0).splitlines() if l.startswith(f'{indent}def '))
    signature = re.match(rf'{indent}def {name}\((self, )?(\w+): [^,)]+(.*)\) -> (.*):$', def_line)
    if signature is None:
        raise RuntimeError(f'generate_pyi: cannot read the signature of {name}: {def_line}')
    self_arg, size_arg, rest, result = signature.groups()
    # After *size the rest is keyword-only already: (*size, *, dtype) is invalid
    rest = rest.removeprefix(', *')
    varargs = f'def {name}({self_arg or ""}*{size_arg}: _int{rest}) -> {result}: ...'
    sized = def_line[len(indent):].rstrip(':') + ': ...'
    stub = f'@typing.overload\n{sized}\n@typing.overload\n{varargs}'
    return replace_definition(text, name, stub, indent)


def tensor_class_bounds(text: str) -> tuple[int, int]:
    start = text.index('\nclass Tensor:\n') + 1
    end = re.compile(r'^\S', re.MULTILINE).search(text, start + len('class Tensor:\n'))
    return start, end.start() if end else len(text)


def postprocess_tensor_stubs(text: str) -> str:
    start, end = tensor_class_bounds(text)
    body = text[start:end]
    for name, stub in TENSOR_METHODS.items():
        body = replace_definition(body, name, stub, '    ')
    for name in TENSOR_SIZE_VARARGS:
        body = add_size_varargs(body, name, '    ')
    text = text[:start] + body + text[end:]
    for name in FUNCTION_SIZE_VARARGS:
        text = add_size_varargs(text, name, '')
    return text


def unwrap_python_methods() -> None:
    """Puts back the pybind11 methods that raxis/__init__.py wrapped in Python
    functions, so pybind11-stubgen reads their C++ signatures instead of
    (*args, **kwargs). Only this process sees the change."""
    import raxis
    for name, attr in list(vars(raxis.Tensor).items()):
        wrapped = getattr(attr, '__wrapped__', None)
        if wrapped is not None:
            setattr(raxis.Tensor, name, wrapped)


def postprocess_stub(stub_path: Path) -> None:
    """Apply all post-processing rules to a generated .pyi file."""
    text = stub_path.read_text()

    # 1. Text replacements (regex if pattern contains \b, plain otherwise)
    for pattern, replacement in TEXT_REPLACEMENTS:
        if r'\b' in pattern:
            text = re.sub(pattern, replacement, text)
        else:
            text = text.replace(pattern, replacement)

    # 2. Stubs of the Tensor methods replaced in Python (after 1, so a shape
    # type is the single word Size)
    if '\nclass Tensor:\n' in text:
        text = postprocess_tensor_stubs(text)

    # 3. Remove unwanted lines
    lines = text.splitlines(keepends=True)
    lines = [l for l in lines if not any(r in l for r in REMOVE_LINES)]

    # 4. Insert extra imports after the last existing import/from line
    insert_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(('import ', 'from ')):
            insert_idx = i + 1

    for j, imp in enumerate(EXTRA_IMPORTS):
        lines.insert(insert_idx + j, imp + '\n')
    insert_idx += len(EXTRA_IMPORTS)

    # 5. Insert type aliases block
    lines.insert(insert_idx, TYPE_ALIASES + '\n')

    stub_path.write_text(''.join(lines))


def main():
    stubs_path = project_path / f'raxis/{cpp_module_name}'

    # Remove previous stubs
    if stubs_path.exists():
        shutil.rmtree(stubs_path)

    unwrap_python_methods()

    # Generate new stubs
    # --exit-code: fail instead of writing stubs with unresolved types
    sys.argv = [None, f'raxis.{cpp_module_name}', '--output', '.', '--exit-code']
    pybind11_stubgen.main()

    # Post-process all generated .pyi files
    for pyi_file in stubs_path.parent.rglob('*.pyi'):
        print(f'Post-processing: {pyi_file.relative_to(project_path)}')
        postprocess_stub(pyi_file)


if __name__ == "__main__":
    main()