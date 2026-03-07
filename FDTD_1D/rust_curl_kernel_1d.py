import ctypes
import platform
from pathlib import Path

import numpy as np


class FDTD1DRustKernel:
    def __init__(self, lib_path: Path):
        self.lib = ctypes.CDLL(str(lib_path))

        self.lib.update_h_interior.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_double,
        ]
        self.lib.update_h_interior.restype = None

        self.lib.update_e_interior.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_double,
        ]
        self.lib.update_e_interior.restype = None

    @staticmethod
    def _is_compatible(*arrs: np.ndarray) -> bool:
        for arr in arrs:
            if arr.dtype != np.float64 or not arr.flags.c_contiguous:
                return False
        return True

    @staticmethod
    def _ptr(arr: np.ndarray):
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def update_h_interior(self, hx, ey, mhx, dz) -> bool:
        if not self._is_compatible(hx, ey, mhx):
            return False
        nz = hx.size
        self.lib.update_h_interior(
            self._ptr(hx),
            self._ptr(ey),
            self._ptr(mhx),
            ctypes.c_size_t(nz),
            ctypes.c_double(dz),
        )
        return True

    def update_e_interior(self, ey, hx, mey, dz) -> bool:
        if not self._is_compatible(ey, hx, mey):
            return False
        nz = ey.size
        self.lib.update_e_interior(
            self._ptr(ey),
            self._ptr(hx),
            self._ptr(mey),
            ctypes.c_size_t(nz),
            ctypes.c_double(dz),
        )
        return True


def _candidate_library_paths() -> list[Path]:
    root = Path(__file__).resolve().parent / "rust_kernel" / "target" / "release"
    system = platform.system().lower()
    if system == "windows":
        names = ["fdtd1d_kernel.dll", "libfdtd1d_kernel.dll"]
    elif system == "darwin":
        names = ["libfdtd1d_kernel.dylib"]
    else:
        names = ["libfdtd1d_kernel.so"]
    return [root / n for n in names]


def load_kernel():
    for p in _candidate_library_paths():
        if p.exists():
            try:
                return FDTD1DRustKernel(p)
            except OSError:
                return None
    return None
