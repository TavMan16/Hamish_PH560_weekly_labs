# Import the setup function used to build Python extension modules.
from setuptools import setup

# Import the cythonize helper used to compile the Cython source file.
from Cython.Build import cythonize


# Build the Cython extension module in place.
setup(
    # Compile the Metropolis kernel source file.
    ext_modules=cythonize("metropolis_kernel.pyx")
)
