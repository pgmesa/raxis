
import sys
import shutil
from pathlib import Path

project_path = Path(__file__).parent.parent.absolute()
if project_path not in sys.path:
    sys.path.append(str(project_path))

import pybind11_stubgen


cpp_module_name = '_C'

    
def main():
    stubs_path = project_path / f'raxis/{cpp_module_name}'
    
    # Remove previous stubs
    if stubs_path.exists():
        shutil.rmtree(stubs_path)
    
    # Generate new stubs
    sys.argv = [None, f'raxis.{cpp_module_name}', '--output', '.']
    pybind11_stubgen.main()


if __name__ == "__main__":
    main()