import ctypes
import platform
from pathlib import Path

import numpy as np


class EzRustCurlKernel:
    def __init__(self, lib_path: Path):
        self.lib = ctypes.CDLL(str(lib_path))

        self.lib.calculate_curl_e_ez.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.calculate_curl_e_ez.restype = None

        self.lib.calculate_curl_h_ez.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.calculate_curl_h_ez.restype = None

    @staticmethod
    def _is_compatible(*arrs: np.ndarray) -> bool:
        for arr in arrs:
            if arr.dtype != np.float64 or not arr.flags.c_contiguous:
                return False
        return True

    @staticmethod
    def _ptr(arr: np.ndarray):
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def calculate_curl_e(self, Ez, d_Ez_x, d_Ez_y, dx, dy, per_x, per_y) -> bool:
        if not self._is_compatible(Ez, d_Ez_x, d_Ez_y):
            return False
        nx, ny = Ez.shape
        self.lib.calculate_curl_e_ez(
            self._ptr(Ez),
            self._ptr(d_Ez_x),
            self._ptr(d_Ez_y),
            ctypes.c_size_t(nx),
            ctypes.c_size_t(ny),
            ctypes.c_double(dx),
            ctypes.c_double(dy),
            ctypes.c_int(1 if per_x else 0),
            ctypes.c_int(1 if per_y else 0),
        )
        return True

    def calculate_curl_h(self, Hx, Hy, d_Hx_y, d_Hy_x, dx, dy, per_x, per_y) -> bool:
        if not self._is_compatible(Hx, Hy, d_Hx_y, d_Hy_x):
            return False
        nx, ny = Hx.shape
        self.lib.calculate_curl_h_ez(
            self._ptr(Hx),
            self._ptr(Hy),
            self._ptr(d_Hx_y),
            self._ptr(d_Hy_x),
            ctypes.c_size_t(nx),
            ctypes.c_size_t(ny),
            ctypes.c_double(dx),
            ctypes.c_double(dy),
            ctypes.c_int(1 if per_x else 0),
            ctypes.c_int(1 if per_y else 0),
        )
        return True


def _candidate_library_paths() -> list[Path]:
    root = Path(__file__).resolve().parent / "rust_kernel" / "target" / "release"
    system = platform.system().lower()
    if system == "windows":
        names = ["fdtd2d_ez_kernel.dll", "libfdtd2d_ez_kernel.dll"]
    elif system == "darwin":
        names = ["libfdtd2d_ez_kernel.dylib"]
    else:
        names = ["libfdtd2d_ez_kernel.so"]
    return [root / n for n in names]


def load_kernel():
    for p in _candidate_library_paths():
        if p.exists():
            try:
                return EzRustCurlKernel(p)
            except OSError:
                return None
    return None
