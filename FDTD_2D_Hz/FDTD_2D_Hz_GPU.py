"""GPU-accelerated variant of :mod:`FDTD_2D_Hz`.

The original ``FDTD_2D_Hz`` solver stores all of its state as NumPy
arrays and performs the Yee update loop with Python for-loops.  This
module provides :class:`FDTD_2D_Hz_GPU`, a drop-in compatible solver
that pushes the field update loop to PyTorch so it can execute on a
GPU (or on a CPU via the same tensor kernels if no GPU is available).

The implementation subclasses :class:`FDTD_2D_Hz` to reuse the rich set
of geometry helpers, source definitions and monitor utilities.  During
``run`` all state arrays are copied to ``torch.Tensor`` instances, the
Yee curls are computed with vectorised tensor operations (which execute
in parallel on the selected device) and the fields are copied back to
NumPy when the simulation finishes.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

from FDTD_2D_Hz import FDTD_2D_Hz


class FDTD_2D_Hz_GPU(FDTD_2D_Hz):
    """GPU enabled solver.

    Parameters are identical to :class:`FDTD_2D_Hz` with two optional
    additions:

    ``device``
        Torch device string.  ``None`` selects ``"cuda"`` when a GPU is
        available and falls back to ``"cpu"`` otherwise.

    ``dtype``
        Torch floating point dtype.  Defaults to ``torch.float32`` which
        keeps memory requirements modest while still matching the
        precision used by the NumPy implementation.
    """

    def __init__(self, *args, device: str | torch.device | None = None,
                 dtype: torch.dtype = torch.float32, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_device(device)
        self.torch_dtype = dtype

    # ------------------------------------------------------------------
    # Helpers
    def set_device(self, device: str | torch.device | None):
        """Update the torch device used by the solver."""

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    # The NumPy solver stores most arrays as ``float64``.  Copy them to
    # tensors with the requested dtype so PyTorch can keep everything on
    # the same device.
    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        data = np.asarray(array, dtype=np.float64)
        tensor = torch.as_tensor(data, dtype=self.torch_dtype, device=self.device)
        return tensor.clone()

    @staticmethod
    def _curl_e(Ex: torch.Tensor, Ey: torch.Tensor, dx: float, dy: float,
                periodic_x: bool, periodic_y: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(d_Ex_y, d_Ey_x)`` for the Yee grid."""

        d_Ex_y = torch.zeros_like(Ex)
        d_Ex_y[:, :-1] = (Ex[:, 1:] - Ex[:, :-1]) / dy
        if periodic_y:
            d_Ex_y[:, -1] = (Ex[:, 0] - Ex[:, -1]) / dy
        else:
            d_Ex_y[:, -1] = (-Ex[:, -1]) / dy

        d_Ey_x = torch.zeros_like(Ey)
        d_Ey_x[:-1, :] = (Ey[1:, :] - Ey[:-1, :]) / dx
        if periodic_x:
            d_Ey_x[-1, :] = (Ey[0, :] - Ey[-1, :]) / dx
        else:
            d_Ey_x[-1, :] = (-Ey[-1, :]) / dx

        return d_Ex_y, d_Ey_x

    @staticmethod
    def _curl_h(Hz: torch.Tensor, dx: float, dy: float,
                periodic_x: bool, periodic_y: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(d_Hz_y, d_Hz_x)`` for the Yee grid."""

        d_Hz_y = torch.zeros_like(Hz)
        d_Hz_y[:, 1:] = (Hz[:, 1:] - Hz[:, :-1]) / dy
        if periodic_y:
            d_Hz_y[:, 0] = (Hz[:, 0] - Hz[:, -1]) / dy
        else:
            d_Hz_y[:, 0] = Hz[:, 0] / dy

        d_Hz_x = torch.zeros_like(Hz)
        d_Hz_x[1:, :] = (Hz[1:, :] - Hz[:-1, :]) / dx
        if periodic_x:
            d_Hz_x[0, :] = (Hz[0, :] - Hz[-1, :]) / dx
        else:
            d_Hz_x[0, :] = Hz[0, :] / dx

        return d_Hz_y, d_Hz_x

    @staticmethod
    def _avg_with_neighbor_torch(arr: torch.Tensor, axis: int,
                                 periodic: bool, direction: int) -> torch.Tensor:
        """Torch version of :meth:`FDTD_2D_Hz._avg_with_neighbor`."""

        if direction not in (-1, 1):
            raise ValueError("direction must be ±1")

        neighbor = torch.zeros_like(arr)
        if axis == 0:
            if direction == -1:
                neighbor[1:, :] = arr[:-1, :]
                neighbor[0, :] = arr[-1, :] if periodic else 0.0
            else:
                neighbor[:-1, :] = arr[1:, :]
                neighbor[-1, :] = arr[0, :] if periodic else 0.0
        elif axis == 1:
            if direction == -1:
                neighbor[:, 1:] = arr[:, :-1]
                neighbor[:, 0] = arr[:, -1] if periodic else 0.0
            else:
                neighbor[:, :-1] = arr[:, 1:]
                neighbor[:, -1] = arr[:, 0] if periodic else 0.0
        else:
            raise ValueError("axis must be 0 or 1")

        return 0.5 * (arr + neighbor)

    # ------------------------------------------------------------------
    # GPU main loop
    def run(self, record_stride: int = 1, is_include_history: bool = True):
        """Execute the simulation using PyTorch tensors."""

        self._init_Coeff()
        self.is_include_history = is_include_history

        Nx, Ny = self.Nx, self.Ny
        per_x = hasattr(self, "periodic") and ('x' in self.periodic)
        per_y = hasattr(self, "periodic") and ('y' in self.periodic)

        # Copy the NumPy arrays to torch.Tensor containers.
        Dx = self._to_tensor(self.Dx)
        Dy = self._to_tensor(self.Dy)
        Ex = self._to_tensor(self.Ex)
        Ey = self._to_tensor(self.Ey)
        Bz = self._to_tensor(self.Bz)
        Hz = self._to_tensor(self.Hz)

        Psi_Dx_y = self._to_tensor(self.Psi_Dx_y)
        Psi_Dy_x = self._to_tensor(self.Psi_Dy_x)
        Psi_Bz_x = self._to_tensor(self.Psi_Bz_x)
        Psi_Bz_y = self._to_tensor(self.Psi_Bz_y)

        kappa_x = self._to_tensor(self.kappa_x)
        kappa_y = self._to_tensor(self.kappa_y)
        ERxx = self._to_tensor(self.ERxx)
        ERyy = self._to_tensor(self.ERyy)
        MRzz = self._to_tensor(self.MRzz)

        b_Dx_y = self._to_tensor(self.b_Dx_y)
        b_Dy_x = self._to_tensor(self.b_Dy_x)
        b_Bz_x = self._to_tensor(self.b_Bz_x)
        b_Bz_y = self._to_tensor(self.b_Bz_y)
        c_Dx_y = self._to_tensor(self.c_Dx_y)
        c_Dy_x = self._to_tensor(self.c_Dy_x)
        c_Bz_x = self._to_tensor(self.c_Bz_x)
        c_Bz_y = self._to_tensor(self.c_Bz_y)

        Hz_prev = Hz.clone()

        self.record_stride = int(record_stride)
        if self.is_include_history:
            Nt_rec = (self.Nt + self.record_stride - 1) // self.record_stride
            self.Nt_rec = int(Nt_rec)
            dtype_hist = self.Ex.dtype
            self.Ex_history = np.zeros((Nt_rec, Nx, Ny), dtype=dtype_hist)
            self.Ey_history = np.zeros((Nt_rec, Nx, Ny), dtype=dtype_hist)
            self.Hz_history = np.zeros((Nt_rec, Nx, Ny), dtype=dtype_hist)
            rec_idx = 0
        else:
            self.Nt_rec = 0
            rec_idx = None

        monitor_results = []
        for m in self.monitors:
            orient = m.get("orientation", "").lower()
            if orient not in ("horizontal", "vertical"):
                continue

            ix0, ix1 = int(m["ix0"]), int(m["ix1"])
            iy0, iy1 = int(m["iy0"]), int(m["iy1"])
            it0, it1 = int(m["it0"]), int(m["it1"])

            if orient == "horizontal":
                L = ix1 - ix0
                Tm = max(0, it1 - it0)
                if L <= 0 or Tm <= 0:
                    continue
                buf = {
                    **m,
                    "Hz": np.empty((Tm, L), dtype=self.Hz.dtype),
                    "Ex": np.empty((Tm, L), dtype=self.Ex.dtype),
                    "Ey": np.empty((Tm, L), dtype=self.Ey.dtype),
                    "_slx": slice(ix0, ix1),
                    "_y": iy0,
                }
            else:
                L = iy1 - iy0
                Tm = max(0, it1 - it0)
                if L <= 0 or Tm <= 0:
                    continue
                buf = {
                    **m,
                    "Hz": np.empty((Tm, L), dtype=self.Hz.dtype),
                    "Ex": np.empty((Tm, L), dtype=self.Ex.dtype),
                    "Ey": np.empty((Tm, L), dtype=self.Ey.dtype),
                    "_x": ix0,
                    "_sly": slice(iy0, iy1),
                }
            monitor_results.append(buf)

        dx, dy = self.dx, self.dy
        M = self.M

        for t_index in tqdm(range(self.Nt), desc="FDTD simulation", unit="step"):
            # Curl(E)
            d_Ex_y, d_Ey_x = self._curl_e(Ex, Ey, dx, dy, per_x, per_y)

            # SF/TF and modal source injections modify the curls.
            for s in self.sources:
                kind = s.get("kind")
                if kind == 'sftf':
                    ix_lo = s["ix0"]
                    ix_hi = s["ix1"]
                    iy_lo = s["iy0"]
                    iy_hi = s["iy1"]

                    nx_side = ix_hi - ix_lo
                    ny_side = iy_hi - iy_lo
                    if nx_side <= 0 or ny_side <= 0:
                        continue

                    t_now = t_index * self.dt
                    kx = float(np.cos(s["angle"]))
                    ky = float(np.sin(s["angle"]))

                    if ix_lo - 1 >= 0:
                        t_edge = t_now - s["Ey_delay_xlo"]
                        Ey_inc = torch.as_tensor(self._g(s, t_edge), dtype=self.torch_dtype, device=self.device)
                        Ey_inc = (kx * Ey_inc)
                        for j_off, j in enumerate(range(iy_lo, iy_hi + 1)):
                            d_Ey_x[ix_lo - 1, j] -= Ey_inc[j_off] / dy

                    if 0 <= ix_hi - 1 < self.Nx:
                        t_edge = t_now - s["Ey_delay_xhi"]
                        Ey_inc = torch.as_tensor(self._g(s, t_edge), dtype=self.torch_dtype, device=self.device)
                        Ey_inc = (kx * Ey_inc)
                        for j_off, j in enumerate(range(iy_lo, iy_hi + 1)):
                            d_Ey_x[ix_hi, j] += Ey_inc[j_off] / dy

                    if iy_lo - 1 >= 0:
                        t_edge = t_now - s["Ex_delay_ylo"]
                        Ex_inc = torch.as_tensor(self._g(s, t_edge), dtype=self.torch_dtype, device=self.device)
                        Ex_inc = (-ky * Ex_inc)
                        for i_off, i in enumerate(range(ix_lo, ix_hi + 1)):
                            d_Ex_y[i, iy_lo - 1] -= Ex_inc[i_off] / dx

                    if 0 <= iy_hi - 1 < self.Ny:
                        t_edge = t_now - s["Ex_delay_yhi"]
                        Ex_inc = torch.as_tensor(self._g(s, t_edge), dtype=self.torch_dtype, device=self.device)
                        Ex_inc = (-ky * Ex_inc)
                        for i_off, i in enumerate(range(ix_lo, ix_hi + 1)):
                            d_Ex_y[i, iy_hi] += Ex_inc[i_off] / dx

                elif kind == 'waveguide-y':
                    y = s["iy0"]
                    i0, i1 = s["ix0"], s["ix1"]
                    lo, hi = (min(i0, i1), max(i0, i1))
                    E_src = torch.tensor(self._g(s, t_index * self.dt), dtype=self.torch_dtype,
                                         device=self.device)
                    for i in range(lo, hi):
                        if 0 <= y - 1 < self.Ny:
                            d_Ex_y[i, y - 1] -= (E_src / dy) * s["Ex_src"][i - lo]

                elif kind == 'waveguide-x':
                    x = s["ix0"]
                    j0, j1 = s["iy0"], s["iy1"]
                    lo, hi = (min(j0, j1), max(j0, j1))
                    E_src = torch.tensor(self._g(s, t_index * self.dt), dtype=self.torch_dtype,
                                         device=self.device)
                    for j in range(lo, hi):
                        if 0 <= x - 1 < self.Nx:
                            d_Ey_x[x - 1, j] -= (E_src / dx) * s["Ey_src"][j - lo]

            # Update magnetic flux and field
            Psi_Bz_x = b_Bz_x * Psi_Bz_x + c_Bz_x * d_Ey_x
            Psi_Bz_y = b_Bz_y * Psi_Bz_y + c_Bz_y * d_Ex_y
            Bz = Bz - M * ((d_Ey_x / kappa_x) - (d_Ex_y / kappa_y) + Psi_Bz_x - Psi_Bz_y)

            # Soft sources directly modify Bz before converting to H
            t_now = t_index * self.dt
            for s in self.sources:
                kind = s.get("kind")
                if kind == 'point':
                    i, j = s["ix0"], s["iy0"]
                    if 0 <= i < self.Nx and 0 <= j < self.Ny:
                        Bz[i, j] += torch.tensor(self._g(s, t_now), dtype=self.torch_dtype, device=self.device)
                elif kind == 'line-soft':
                    val = torch.tensor(self._g(s, t_now), dtype=self.torch_dtype, device=self.device)
                    if s["ix0"] != s["ix1"]:
                        y = s["iy0"]
                        lo = min(s["ix0"], s["ix1"])
                        hi = max(s["ix0"], s["ix1"])
                        if 0 <= y < self.Ny:
                            for i in range(lo, hi):
                                if 0 <= i < self.Nx:
                                    Bz[i, y] += val
                    else:
                        x = s["ix0"]
                        lo = min(s["iy0"], s["iy1"])
                        hi = max(s["iy0"], s["iy1"])
                        if 0 <= x < self.Nx:
                            for j in range(lo, hi):
                                if 0 <= j < self.Ny:
                                    Bz[x, j] += val

            Hz = Bz / MRzz

            # Curl(H)
            d_Hz_y, d_Hz_x = self._curl_h(Hz, dx, dy, per_x, per_y)

            # Inject magnetic field waveguide sources
            for s in self.sources:
                kind = s.get("kind")
                if kind == 'sftf':
                    ix_lo = s["ix0"]
                    ix_hi = s["ix1"]
                    iy_lo = s["iy0"]
                    iy_hi = s["iy1"]

                    nx_side = ix_hi - ix_lo
                    ny_side = iy_hi - iy_lo
                    if nx_side <= 0 or ny_side <= 0:
                        continue

                    t_half = (t_index + 0.5) * self.dt
                    if ix_lo < self.Nx:
                        t_edge = t_half - s["Hz_delay_xlo"]
                        Hz_inc = torch.as_tensor(self._g(s, t_edge), dtype=self.torch_dtype, device=self.device)
                        for j_off, j in enumerate(range(iy_lo, iy_hi + 1)):
                            if 0 <= j < self.Ny:
                                d_Hz_x[ix_lo, j] -= Hz_inc[j_off] / dx

                    if ix_hi < self.Nx:
                        t_edge = t_half - s["Hz_delay_xhi"]
                        Hz_inc = torch.as_tensor(self._g(s, t_edge), dtype=self.torch_dtype, device=self.device)
                        idx = ix_hi + 1
                        if idx < self.Nx:
                            for j_off, j in enumerate(range(iy_lo, iy_hi + 1)):
                                if 0 <= j < self.Ny:
                                    d_Hz_x[idx, j] += Hz_inc[j_off] / dx

                    if iy_lo < self.Ny:
                        t_edge = t_half - s["Hz_delay_ylo"]
                        Hz_inc = torch.as_tensor(self._g(s, t_edge), dtype=self.torch_dtype, device=self.device)
                        for i_off, i in enumerate(range(ix_lo, ix_hi + 1)):
                            if 0 <= i < self.Nx:
                                d_Hz_y[i, iy_lo] -= Hz_inc[i_off] / dy

                    if iy_hi < self.Ny:
                        t_edge = t_half - s["Hz_delay_yhi"]
                        Hz_inc = torch.as_tensor(self._g(s, t_edge), dtype=self.torch_dtype, device=self.device)
                        idx = iy_hi + 1
                        if idx < self.Ny:
                            for i_off, i in enumerate(range(ix_lo, ix_hi + 1)):
                                if 0 <= i < self.Nx:
                                    d_Hz_y[i, idx] += Hz_inc[i_off] / dy

                elif kind == 'waveguide-y':
                    n_eff = s["n_eff"]
                    H_src = -self._g(s, t_index * self.dt + self.dy * n_eff / (2 * self.c0) + self.dt / 2.0)
                    H_src = torch.tensor(H_src, dtype=self.torch_dtype, device=self.device)
                    y = s["iy0"]
                    i0, i1 = s["ix0"], s["ix1"]
                    lo, hi = (min(i0, i1), max(i0, i1))
                    for i in range(lo, hi):
                        if 0 <= y < self.Ny:
                            d_Hz_y[i, y] += (H_src / dy) * s["Hz_src"][i - lo]

                elif kind == 'waveguide-x':
                    n_eff = s["n_eff"]
                    H_src = -self._g(s, t_index * self.dt + self.dx * n_eff / (2 * self.c0) + self.dt / 2.0)
                    H_src = torch.tensor(H_src, dtype=self.torch_dtype, device=self.device)
                    x = s["ix0"]
                    j0, j1 = s["iy0"], s["iy1"]
                    lo, hi = (min(j0, j1), max(j0, j1))
                    for j in range(lo, hi):
                        if 0 <= x < self.Nx:
                            d_Hz_x[x, j] -= (H_src / dx) * s["Hz_src"][j - lo]

            Psi_Dx_y = b_Dx_y * Psi_Dx_y + c_Dx_y * d_Hz_y
            Psi_Dy_x = b_Dy_x * Psi_Dy_x + c_Dy_x * d_Hz_x
            Dx = Dx + M * (d_Hz_y / kappa_y + Psi_Dx_y)
            Dy = Dy - M * (d_Hz_x / kappa_x + Psi_Dy_x)
            Ex = Dx / ERxx
            Ey = Dy / ERyy

            if monitor_results:
                Hz_center = 0.5 * (Hz + Hz_prev)
                Ex_center = self._avg_with_neighbor_torch(Ex, axis=1, periodic=per_y, direction=+1)
                Ey_center = self._avg_with_neighbor_torch(Ey, axis=0, periodic=per_x, direction=+1)

            for buf in monitor_results:
                if buf["it0"] <= t_index < buf["it1"]:
                    k = t_index - buf["it0"]
                    if buf["orientation"] == "horizontal":
                        hz_slice = Hz_center[buf["_slx"], buf["_y"]]
                        ex_slice = Ex_center[buf["_slx"], buf["_y"]]
                        ey_slice = Ey_center[buf["_slx"], buf["_y"]]
                    else:
                        hz_slice = Hz_center[buf["_x"], buf["_sly"]]
                        ex_slice = Ex_center[buf["_x"], buf["_sly"]]
                        ey_slice = Ey_center[buf["_x"], buf["_sly"]]

                    buf["Hz"][k, :] = hz_slice.detach().cpu().numpy()
                    buf["Ex"][k, :] = ex_slice.detach().cpu().numpy()
                    buf["Ey"][k, :] = ey_slice.detach().cpu().numpy()

            if monitor_results:
                Hz_prev = Hz.clone()

            if self.is_include_history and (t_index % self.record_stride) == 0:
                self.Ex_history[rec_idx, :, :] = Ex.detach().cpu().numpy()
                self.Ey_history[rec_idx, :, :] = Ey.detach().cpu().numpy()
                self.Hz_history[rec_idx, :, :] = Hz.detach().cpu().numpy()
                rec_idx += 1

        self.monitor_results = []
        for buf in monitor_results:
            out = {k: v for k, v in buf.items() if not k.startswith("_")}
            self.monitor_results.append(out)

        # Copy tensors back to NumPy so the rest of the API keeps the
        # same behaviour as the CPU-only solver.
        self.Dx = Dx.detach().cpu().numpy()
        self.Dy = Dy.detach().cpu().numpy()
        self.Ex = Ex.detach().cpu().numpy()
        self.Ey = Ey.detach().cpu().numpy()
        self.Bz = Bz.detach().cpu().numpy()
        self.Hz = Hz.detach().cpu().numpy()

        self.Psi_Dx_y = Psi_Dx_y.detach().cpu().numpy()
        self.Psi_Dy_x = Psi_Dy_x.detach().cpu().numpy()
        self.Psi_Bz_x = Psi_Bz_x.detach().cpu().numpy()
        self.Psi_Bz_y = Psi_Bz_y.detach().cpu().numpy()
