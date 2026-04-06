from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        [
            "metropolis_kernel.pyx",
            "xy_metropolis_kernel.pyx",
        ]
    )
)
