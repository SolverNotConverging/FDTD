"""Build native FDTD runtimes and optional curl kernels in place."""

from pathlib import Path
import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup


ROOT = Path(__file__).resolve().parent

extensions = [
    Extension("FDTD_common._compiled_2d", [str(ROOT / "FDTD_common/compiled_2d.pyx")],
              include_dirs=[np.get_include()],
              libraries=[] if sys.platform == 'win32' else ['dl']),
    Extension("FDTD_1D._cython_kernel_1d", [str(ROOT / "FDTD_1D" / "cython_kernel_1d.pyx")],
              include_dirs=[np.get_include()]),
    Extension("FDTD_2D_Ez._cython_kernel_ez", [str(ROOT / "FDTD_2D_Ez/cython_kernel_ez.pyx")],
              include_dirs=[np.get_include()]),
    Extension("FDTD_2D_Hz._cython_kernel_hz", [str(ROOT / "FDTD_2D_Hz/cython_kernel_hz.pyx")],
              include_dirs=[np.get_include()]),
    Extension("FDTD_2D_GR._cython_kernel_gr", [str(ROOT / "FDTD_2D_GR/cython_kernel_gr.pyx")],
              include_dirs=[np.get_include()]),
    Extension("FDTD_3D._cython_kernel_3d", [str(ROOT / "FDTD_3D" / "cython_kernel_3d.pyx")],
              include_dirs=[np.get_include()]),
]

setup(
    name="fdtd-cython-kernels",
    version="0.3.0",
    packages=find_packages(include=["FDTD_1D*", "FDTD_2D_Ez*", "FDTD_2D_Hz*", "FDTD_2D_GR*", "legacy*", "FDTD_3D*", "FDTD_common*"]),
    ext_modules=cythonize(extensions, language_level=3,
                          build_dir=str(ROOT / "build" / "cython")),
)
