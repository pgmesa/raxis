"""The raxis-cuda12 / raxis-cuda13 wheels: libraxis_cuda for one CUDA major
version, built first by `scripts/build_libraxis.py --cuda-plugin`.

The wheel's name follows the toolkit the library was built with (the
cuda_version.txt the build script writes), so a CUDA 13 build cannot end up
in raxis-cuda12. No [project] table in pyproject.toml: the name of a
distribution cannot be dynamic there.
"""
import re
from pathlib import Path

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.dist import Distribution


plugin_path = Path(__file__).parent.absolute()
project_path = plugin_path.parent.parent

version = re.search(r'^__version__ = "(.+)"$', (project_path / 'raxis' / '__init__.py').read_text(), re.M).group(1)
cuda_version = (plugin_path / 'raxis_cuda' / 'lib' / 'cuda_version.txt').read_text().strip()
cuda_major = int(cuda_version.split('.')[0])

# NVIDIA renamed the CUDA 13 wheels: nvidia-cublas (13.x) instead of nvidia-cublas-cu13
cublas = {12: 'nvidia-cublas-cu12>=12.1', 13: 'nvidia-cublas>=13.0,<14'}[cuda_major]


class BinaryDistribution(Distribution):
    """Native code: installed into platlib, next to raxis (libraxis_cuda's
    RUNPATH finds libraxis at ../../raxis/lib)."""

    def has_ext_modules(self):
        return True


class PlatformWheel(bdist_wheel):
    """Tagged py3-none-<platform>: the library does not use Python's C API,
    so one wheel serves every Python version (as JAX's CUDA plugins)."""

    def get_tag(self):
        return 'py3', 'none', super().get_tag()[2]


package = f'raxis_cuda{cuda_major}'
setup(
    name=f'raxis-cuda{cuda_major}',
    version=version,
    description=f"CUDA {cuda_major} backend of raxis (NVIDIA GPUs)",
    long_description=(plugin_path / 'README.md').read_text().replace('{cuda_major}', str(cuda_major)),
    long_description_content_type='text/markdown',
    url='https://github.com/pgmesa/raxis',
    author='Pablo García Mesa',
    author_email='pgmesa.sm@gmail.com',
    license='MIT',
    license_files=['LICENSE', 'THIRD_PARTY_NOTICES.txt'],
    python_requires='>=3.10,<3.15',
    # The exact version: libraxis_cuda links against that libraxis
    install_requires=[f'raxis=={version}', cublas],
    packages=[package],
    package_dir={package: 'raxis_cuda'},
    package_data={package: ['lib/*']},
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: GPU :: NVIDIA CUDA',
        f'Environment :: GPU :: NVIDIA CUDA :: {cuda_major}',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
    ],
    distclass=BinaryDistribution,
    cmdclass={'bdist_wheel': PlatformWheel},
)
