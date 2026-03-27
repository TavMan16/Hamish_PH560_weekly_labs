"""Build script for the Cython walk kernel."""

from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np


extensions = [
    Extension(
        "walk_kernel",
        ["walk_kernel.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="walk_kernel",
    ext_modules=cythonize(extensions, language_level="3"),
)
