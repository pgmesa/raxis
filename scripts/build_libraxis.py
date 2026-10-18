"""Build libraxis with CMake and install it into the raxis package.

The install step fills raxis/lib (shared libraries and the _C extension),
raxis/include and raxis/share, then the type stubs are regenerated. The
package is then ready to import from the source tree or to be packaged as a
wheel (`uv build --wheel`), which is what the CI does for each Python version.

Run it with the Python the extension is for: it builds against that
interpreter and finds MKL in its environment (`uv sync --extra build`).
"""
import sys
import shutil
import argparse
import platform
import subprocess
import sysconfig
from pathlib import Path


project_path = Path(__file__).parent.parent.absolute()
package_path = project_path / 'raxis'
libraxis_path = project_path / 'libraxis'


def run(*args):
    print('[+]', ' '.join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], check=True, cwd=project_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--build-type', default='Release', choices=['Release', 'Debug', 'RelWithDebInfo'])
    parser.add_argument('--no-stubs', action='store_true', help="Skip the .pyi generation")
    args = parser.parse_args()

    # Leftovers of another build (e.g. _C for another Python version) would
    # end up in the wheel
    for name in ('lib', 'include', 'share', '_C'):
        shutil.rmtree(package_path / name, ignore_errors=True)

    # pip installs MKL's CMake config and libraries under <prefix>/Library on
    # Windows and directly under <prefix> elsewhere
    prefix = Path(sys.prefix)
    if platform.system() == 'Windows':
        prefix = prefix / 'Library'

    tag = sysconfig.get_config_var('EXT_SUFFIX').split('.')[1]
    build_path = libraxis_path / 'build' / f'{tag}-{args.build_type.lower()}'

    configure = [
        'cmake', '-S', libraxis_path, '-B', build_path,
        f'-DCMAKE_BUILD_TYPE={args.build_type}',
        f'-DCMAKE_INSTALL_PREFIX={package_path}',
        # raxis/__init__.py loads from raxis/lib; GNUInstallDirs would pick
        # lib64 on RHEL-based systems such as the manylinux images
        '-DCMAKE_INSTALL_LIBDIR=lib',
        f'-DCMAKE_PREFIX_PATH={prefix}',
        f'-DPython_EXECUTABLE={sys.executable}',
    ]
    if shutil.which('ninja') and platform.system() != 'Windows':
        configure += ['-G', 'Ninja']

    run(*configure)
    build = ['cmake', '--build', build_path, '--config', args.build_type]
    # SLEEF generates headers shared by several of its targets, which races
    # under Visual Studio's parallel project builds: build it serially first
    run(*build, '--target', 'sleef')
    run(*build, '--target', 'install', '--parallel')

    if not args.no_stubs:
        run(sys.executable, project_path / 'scripts' / 'generate_pyi.py')


if __name__ == '__main__':
    main()
