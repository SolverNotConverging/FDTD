import ctypes
import platform
from pathlib import Path

import numpy as np


class HzRustCurlKernel:
    def __init__(self, lib_path: Path):
        self.lib = ctypes.CDLL(str(lib_path))

        self.lib.calculate_curl_e_hz.argtypes = [
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
        self.lib.calculate_curl_e_hz.restype = None

        self.lib.calculate_curl_h_hz.argtypes = [
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
        self.lib.calculate_curl_h_hz.restype = None

    @staticmethod
    def _is_compatible(*arrs: np.ndarray) -> bool:
        for arr in arrs:
            if arr.dtype != np.float64 or not arr.flags.c_contiguous:
                return False
        return True

    @staticmethod
    def _ptr(arr: np.ndarray):
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def calculate_curl_e(self, Ex, Ey, d_Ex_y, d_Ey_x, dx, dy, per_x, per_y) -> bool:
        if not self._is_compatible(Ex, Ey, d_Ex_y, d_Ey_x):
            return False
        nx, ny = Ex.shape
        self.lib.calculate_curl_e_hz(
            self._ptr(Ex),
            self._ptr(Ey),
            self._ptr(d_Ex_y),
            self._ptr(d_Ey_x),
            ctypes.c_size_t(nx),
            ctypes.c_size_t(ny),
            ctypes.c_double(dx),
            ctypes.c_double(dy),
            ctypes.c_int(1 if per_x else 0),
            ctypes.c_int(1 if per_y else 0),
        )
        return True

    def calculate_curl_h(self, Hz, d_Hz_y, d_Hz_x, dx, dy, per_x, per_y) -> bool:
        if not self._is_compatible(Hz, d_Hz_y, d_Hz_x):
            return False
        nx, ny = Hz.shape
        self.lib.calculate_curl_h_hz(
            self._ptr(Hz),
            self._ptr(d_Hz_y),
            self._ptr(d_Hz_x),
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
        names = ["fdtd2d_hz_kernel.dll", "libfdtd2d_hz_kernel.dll"]
    elif system == "darwin":
        names = ["libfdtd2d_hz_kernel.dylib"]
    else:
        names = ["libfdtd2d_hz_kernel.so"]
    return [root / n for n in names]


def load_kernel():
    for p in _candidate_library_paths():
        if p.exists():
            try:
                return HzRustCurlKernel(p)
            except OSError:
                return None
    return None
