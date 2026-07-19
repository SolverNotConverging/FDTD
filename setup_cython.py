"""Build all optional Cython curl kernels in place."""

from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup


ROOT = Path(__file__).resolve().parent

extensions = [
    Extension("FDTD_1D._cython_kernel_1d", [str(ROOT / "FDTD_1D" / "cython_kernel_1d.pyx")],
              include_dirs=[np.get_include()]),
    Extension("FDTD_2D_Ez._cython_kernel_ez", [str(ROOT / "FDTD_2D_Ez" / "cython_kernel_ez.pyx")],
              include_dirs=[np.get_include()]),
    Extension("FDTD_2D_Hz._cython_kernel_hz", [str(ROOT / "FDTD_2D_Hz" / "cython_kernel_hz.pyx")],
              include_dirs=[np.get_include()]),
    Extension("FDTD_3D._cython_kernel_3d", [str(ROOT / "FDTD_3D" / "cython_kernel_3d.pyx")],
              include_dirs=[np.get_include()]),
]

setup(
    name="fdtd-cython-kernels",
    version="0.3.0",
    packages=["FDTD_1D", "FDTD_2D_Ez", "FDTD_2D_Hz", "FDTD_3D"],
    ext_modules=cythonize(extensions, language_level=3,
                          build_dir=str(ROOT / "build" / "cython")),
)
