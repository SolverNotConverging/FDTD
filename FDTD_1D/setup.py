"""Build the optional Cython update kernel in place.

Run ``python FDTD_1D/setup.py build_ext --inplace`` from the repository root.
The solver imports the resulting extension when present and otherwise uses its
pure-Python loops.
"""

import os
from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup


HERE = Path(__file__).resolve().parent
os.chdir(HERE.parent)

extensions = [
    Extension(
        "FDTD_1D._cython_kernel_1d",
        [str(HERE / "cython_kernel_1d.pyx")],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="fdtd-1d-cython-kernel",
    version="0.1.0",
    packages=["FDTD_1D"],
    ext_modules=cythonize(extensions, language_level=3),
)
