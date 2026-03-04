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


def postprocess_stub(stub_path: Path) -> None:
    """Apply all post-processing rules to a generated .pyi file."""
    text = stub_path.read_text()

    # 1. Text replacements (regex if pattern contains \b, plain otherwise)
    for pattern, replacement in TEXT_REPLACEMENTS:
        if r'\b' in pattern:
            text = re.sub(pattern, replacement, text)
        else:
            text = text.replace(pattern, replacement)

    # 2. Remove unwanted lines
    lines = text.splitlines(keepends=True)
    lines = [l for l in lines if not any(r in l for r in REMOVE_LINES)]

    # 3. Insert extra imports after the last existing import/from line
    insert_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(('import ', 'from ')):
            insert_idx = i + 1

    for j, imp in enumerate(EXTRA_IMPORTS):
        lines.insert(insert_idx + j, imp + '\n')
    insert_idx += len(EXTRA_IMPORTS)

    # 4. Insert type aliases block
    lines.insert(insert_idx, TYPE_ALIASES + '\n')

    stub_path.write_text(''.join(lines))


def main():
    stubs_path = project_path / f'raxis/{cpp_module_name}'

    # Remove previous stubs
    if stubs_path.exists():
        shutil.rmtree(stubs_path)

    # Generate new stubs
    sys.argv = [None, f'raxis.{cpp_module_name}', '--output', '.']
    pybind11_stubgen.main()

    # Post-process all generated .pyi files
    for pyi_file in stubs_path.parent.rglob('*.pyi'):
        print(f'Post-processing: {pyi_file.relative_to(project_path)}')
        postprocess_stub(pyi_file)


if __name__ == "__main__":
    main()