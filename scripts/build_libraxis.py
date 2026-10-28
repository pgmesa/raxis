"""Build libraxis with CMake and install it into the raxis package.

The install step fills raxis/lib (shared libraries and the _C extension),
raxis/include and raxis/share, then the type stubs are regenerated. The
package is then ready to import from the source tree or to be packaged as a
wheel (`uv build --wheel`), which is what the CI does for each Python version.

With --cuda-plugin it builds the CUDA backend for the raxis-cuda12 /
raxis-cuda13 wheels instead (plugins/cuda): libraxis_cuda goes to
plugins/cuda/raxis_cuda/lib and raxis/ is left as it is. The library does not
link Python, so one build serves every Python version.

Run it with the Python the extension is for: it builds against that
interpreter and finds MKL in its environment (`uv sync --extra build`).
"""
import os
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
cuda_plugin_lib_path = project_path / 'plugins' / 'cuda' / 'raxis_cuda' / 'lib'


def run(*args):
    print('[+]', ' '.join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], check=True, cwd=project_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--build-type', default='Release', choices=['Release', 'Debug', 'RelWithDebInfo'])
    parser.add_argument('--no-stubs', action='store_true', help="Skip the .pyi generation")
    parser.add_argument('--cuda', action='store_true', help="Also build libraxis_cuda (needs the CUDA toolkit)")
    parser.add_argument('--cuda-plugin', action='store_true',
                        help="Build only libraxis_cuda, into plugins/cuda/raxis_cuda/lib (raxis/ is not touched)")
    args = parser.parse_args()

    # Leftovers of another build (e.g. _C for another Python version) would
    # end up in the wheel
    install_path = package_path
    if args.cuda_plugin:
        install_path = libraxis_path / 'build' / 'cuda-plugin-install'
        shutil.rmtree(install_path, ignore_errors=True)
        shutil.rmtree(cuda_plugin_lib_path, ignore_errors=True)
    else:
        for name in ('lib', 'include', 'share', '_C'):
            shutil.rmtree(package_path / name, ignore_errors=True)

    # pip installs MKL's CMake config and libraries under <prefix>/Library on
    # Windows and directly under <prefix> elsewhere. RAXIS_MKL_PREFIX points
    # to an MKL installed outside this environment (`pip install --prefix`),
    # which CI shares across Python versions
    prefix = Path(os.environ.get('RAXIS_MKL_PREFIX', sys.prefix))
    if platform.system() == 'Windows':
        prefix = prefix / 'Library'

    tag = 'cuda-plugin' if args.cuda_plugin else sysconfig.get_config_var('EXT_SUFFIX').split('.')[1]
    build_path = libraxis_path / 'build' / f'{tag}-{args.build_type.lower()}'

    configure = [
        'cmake', '-S', libraxis_path, '-B', build_path,
        f'-DCMAKE_BUILD_TYPE={args.build_type}',
        f'-DCMAKE_INSTALL_PREFIX={install_path}',
        # raxis/__init__.py loads from raxis/lib; GNUInstallDirs would pick
        # lib64 on RHEL-based systems such as the manylinux images
        '-DCMAKE_INSTALL_LIBDIR=lib',
        f'-DCMAKE_PREFIX_PATH={prefix}',
        f'-DPython_EXECUTABLE={sys.executable}',
        f'-DRAXIS_USE_CUDA={"ON" if args.cuda or args.cuda_plugin else "OFF"}',
    ]
    if args.cuda_plugin:
        # libraxis itself is built (libraxis_cuda links it) but not shipped:
        # the plugin loads the one of the raxis wheel of the same version
        configure += ['-DBUILD_PYTHON_BINDINGS=OFF']
    if shutil.which('ninja') and platform.system() != 'Windows':
        configure += ['-G', 'Ninja']

    run(*configure)
    build = ['cmake', '--build', build_path, '--config', args.build_type]
    # SLEEF generates headers shared by several of its targets, which races
    # under Visual Studio's parallel project builds: build it serially first
    run(*build, '--target', 'sleef')
    # A bare --parallel is `make -j` without a limit under Makefiles (the
    # manylinux images have no Ninja): every compile at once runs CI out of memory
    run(*build, '--target', 'install', '--parallel', str(os.cpu_count() or 1))

    if args.cuda_plugin:
        cuda_plugin_lib_path.mkdir(parents=True)
        name = 'raxis_cuda.dll' if platform.system() == 'Windows' else 'libraxis_cuda.so'
        shutil.copy2(install_path / 'lib' / name, cuda_plugin_lib_path / name)
        # The toolkit version names the wheel (raxis-cuda12 / raxis-cuda13)
        cuda_version = (build_path / 'cuda_version.txt').read_text().strip()
        shutil.copy2(build_path / 'cuda_version.txt', cuda_plugin_lib_path / 'cuda_version.txt')
        for notice in ('LICENSE', 'THIRD_PARTY_NOTICES.txt'):
            shutil.copy2(project_path / notice, cuda_plugin_lib_path.parent.parent / notice)
        print(f'[+] {cuda_plugin_lib_path / name} (CUDA {cuda_version})')
    elif not args.no_stubs:
        run(sys.executable, project_path / 'scripts' / 'generate_pyi.py')


if __name__ == '__main__':
    main()
