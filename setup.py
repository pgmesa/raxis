from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    """The package ships the shared libraries built by CMake (raxis/lib), so
    the wheel must be tagged for one Python version and platform."""

    def has_ext_modules(self):
        return True


setup(distclass=BinaryDistribution)
