"""GPU-accelerated variant of :mod:`FDTD_2D_Ez`."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

from FDTD_2D_Ez import FDTD_2D_Ez


class FDTD_2D_Ez_GPU(FDTD_2D_Ez):
    """Run the 2-D TMz solver on PyTorch tensors."""

    def __init__(self, *args, device: str | torch.device | None = None,
                 dtype: torch.dtype = torch.float32, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_device(device)
        self.torch_dtype = dtype

    # ------------------------------------------------------------------
    # Helpers
    def set_device(self, device: str | torch.device | None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        data = np.asarray(array, dtype=np.float64)
        tensor = torch.as_tensor(data, dtype=self.torch_dtype, device=self.device)
        return tensor.clone()

    @staticmethod
    def _curl_e(Ez: torch.Tensor, dx: float, dy: float,
                periodic_x: bool, periodic_y: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (d_Ez_y, d_Ez_x)."""

        d_Ez_y = torch.zeros_like(Ez)
        d_Ez_y[:, :-1] = (Ez[:, 1:] - Ez[:, :-1]) / dy
        if periodic_y:
            d_Ez_y[:, -1] = (Ez[:, 0] - Ez[:, -1]) / dy
        else:
            d_Ez_y[:, -1] = (-Ez[:, -1]) / dy

        d_Ez_x = torch.zeros_like(Ez)
        d_Ez_x[:-1, :] = (Ez[1:, :] - Ez[:-1, :]) / dx
        if periodic_x:
            d_Ez_x[-1, :] = (Ez[0, :] - Ez[-1, :]) / dx
        else:
            d_Ez_x[-1, :] = (-Ez[-1, :]) / dx

        return d_Ez_y, d_Ez_x

    @staticmethod
    def _curl_h(Hx: torch.Tensor, Hy: torch.Tensor, dx: float, dy: float,
                periodic_x: bool, periodic_y: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (d_Hx_y, d_Hy_x)."""

        d_Hx_y = torch.zeros_like(Hx)
        d_Hx_y[:, 1:] = (Hx[:, 1:] - Hx[:, :-1]) / dy
        if periodic_y:
            d_Hx_y[:, 0] = (Hx[:, 0] - Hx[:, -1]) / dy
        else:
            d_Hx_y[:, 0] = (Hx[:, 0] - 0.0) / dy

        d_Hy_x = torch.zeros_like(Hy)
        d_Hy_x[1:, :] = (Hy[1:, :] - Hy[:-1, :]) / dx
        if periodic_x:
            d_Hy_x[0, :] = (Hy[0, :] - Hy[-1, :]) / dx
        else:
            d_Hy_x[0, :] = (Hy[0, :] - 0.0) / dx

        return d_Hx_y, d_Hy_x

    @staticmethod
    def _avg_with_neighbor_torch(arr: torch.Tensor, axis: int,
                                 periodic: bool, direction: int) -> torch.Tensor:
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
    def run(self, record_stride: int = 1, is_include_history: bool = True):
        self._init_Coeff()
        self.is_include_history = is_include_history

        Nx, Ny = self.Nx, self.Ny
        per_x = hasattr(self, "periodic") and ('x' in self.periodic)
        per_y = hasattr(self, "periodic") and ('y' in self.periodic)

        Bx = self._to_tensor(self.Bx)
        By = self._to_tensor(self.By)
        Hx = self._to_tensor(self.Hx)
        Hy = self._to_tensor(self.Hy)
        Dz = self._to_tensor(self.Dz)
        Ez = self._to_tensor(self.Ez)

        Psi_Bx_y = self._to_tensor(self.Psi_Bx_y)
        Psi_By_x = self._to_tensor(self.Psi_By_x)
        Psi_Dz_x = self._to_tensor(self.Psi_Dz_x)
        Psi_Dz_y = self._to_tensor(self.Psi_Dz_y)

        kappa_x = self._to_tensor(self.kappa_x)
        kappa_y = self._to_tensor(self.kappa_y)
        ERzz = self._to_tensor(self.ERzz)
        MRxx = self._to_tensor(self.MRxx)
        MRyy = self._to_tensor(self.MRyy)

        b_Bx_y = self._to_tensor(self.b_Bx_y)
        b_By_x = self._to_tensor(self.b_By_x)
        b_Dz_x = self._to_tensor(self.b_Dz_x)
        b_Dz_y = self._to_tensor(self.b_Dz_y)

        c_Bx_y = self._to_tensor(self.c_Bx_y)
        c_By_x = self._to_tensor(self.c_By_x)
        c_Dz_x = self._to_tensor(self.c_Dz_x)
        c_Dz_y = self._to_tensor(self.c_Dz_y)

        Hx_prev = Hx.clone()
        Hy_prev = Hy.clone()

        self.record_stride = int(record_stride)

        if self.is_include_history:
            Nt_rec = (self.Nt + self.record_stride - 1) // self.record_stride
            self.Nt_rec = int(Nt_rec)
            dtype_hist = self.Hx.dtype
            self.Hx_history = np.zeros((Nt_rec, Nx, Ny), dtype=dtype_hist)
            self.Hy_history = np.zeros((Nt_rec, Nx, Ny), dtype=dtype_hist)
            self.Ez_history = np.zeros((Nt_rec, Nx, Ny), dtype=dtype_hist)
            self.Dz_history = np.zeros((Nt_rec, Nx, Ny), dtype=dtype_hist)
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
                    "Ez": np.empty((Tm, L), dtype=self.Ez.dtype),
                    "Hx": np.empty((Tm, L), dtype=self.Hx.dtype),
                    "Hy": np.empty((Tm, L), dtype=self.Hy.dtype),
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
                    "Ez": np.empty((Tm, L), dtype=self.Ez.dtype),
                    "Hx": np.empty((Tm, L), dtype=self.Hx.dtype),
                    "Hy": np.empty((Tm, L), dtype=self.Hy.dtype),
                    "_x": ix0,
                    "_sly": slice(iy0, iy1),
                }
            monitor_results.append(buf)

        dx, dy = self.dx, self.dy
        M = self.M

        d_Ez_y = torch.zeros_like(Ez)
        d_Ez_x = torch.zeros_like(Ez)
        d_Hx_y = torch.zeros_like(Hx)
        d_Hy_x = torch.zeros_like(Hy)

        for t_index in tqdm(range(self.Nt), desc="FDTD simulation", unit="step"):
            d_Ez_y, d_Ez_x = self._curl_e(Ez, dx, dy, per_x, per_y)

            for s in self.sources:
                kind = s.get("kind")
                if kind == 'sftf':
                    ix_lo = s["ix0"]
                    ix_hi = s["ix1"]
                    iy_lo = s["iy0"]
                    iy_hi = s["iy1"]

                    t_now = t_index * self.dt

                    if ix_lo - 1 >= 0:
                        t_edge = t_now - s["Ez_delay_xlo"]
                        Ezsrc_xlo = torch.as_tensor(self._g(s, t_edge), dtype=self.torch_dtype, device=self.device)
                        for j_off, j in enumerate(range(iy_lo, iy_hi + 1)):
                            d_Ez_x[ix_lo - 1, j] -= Ezsrc_xlo[j_off] / dx

                    if ix_hi - 1 >= 0 and ix_hi - 1 < self.Nx:
                        t_edge = t_now - s["Ez_delay_xhi"]
                        Ezsrc_xhi = torch.as_tensor(self._g(s, t_edge), dtype=self.torch_dtype, device=self.device)
                        for j_off, j in enumerate(range(iy_lo, iy_hi + 1)):
                            d_Ez_x[ix_hi, j] += Ezsrc_xhi[j_off] / dx

                    if iy_lo - 1 >= 0:
                        t_edge = t_now - s["Ez_delay_ylo"]
                        Ezsrc_ylo = torch.as_tensor(self._g(s, t_edge), dtype=self.torch_dtype, device=self.device)
                        for i_off, i in enumerate(range(ix_lo, ix_hi + 1)):
                            d_Ez_y[i, iy_lo - 1] -= Ezsrc_ylo[i_off] / dy

                    if iy_hi - 1 >= 0 and iy_hi - 1 < self.Ny:
                        t_edge = t_now - s["Ez_delay_yhi"]
                        Ezsrc_yhi = torch.as_tensor(self._g(s, t_edge), dtype=self.torch_dtype, device=self.device)
                        for i_off, i in enumerate(range(ix_lo, ix_hi + 1)):
                            d_Ez_y[i, iy_hi] += Ezsrc_yhi[i_off] / dy

                elif kind == 'waveguide-y':
                    y = s["iy0"]
                    i0, i1 = s["ix0"], s["ix1"]
                    lo, hi = (min(i0, i1), max(i0, i1))
                    E_src = torch.as_tensor(self._g(s, t_index * self.dt), dtype=self.torch_dtype, device=self.device)
                    for i in range(lo, hi):
                        if 0 <= y - 1 < self.Ny:
                            d_Ez_y[i, y - 1] -= (E_src / dy) * float(s["Ez_src"][i - lo])
                elif kind == 'waveguide-x':
                    x = s["ix0"]
                    j0, j1 = s["iy0"], s["iy1"]
                    lo, hi = (min(j0, j1), max(j0, j1))
                    E_src = torch.as_tensor(self._g(s, t_index * self.dt), dtype=self.torch_dtype, device=self.device)
                    for j in range(lo, hi):
                        if 0 <= x - 1 < self.Nx:
                            d_Ez_x[x - 1, j] -= (E_src / dx) * float(s["Ez_src"][j - lo])

            Psi_Bx_y = b_Bx_y * Psi_Bx_y + c_Bx_y * d_Ez_y
            Psi_By_x = b_By_x * Psi_By_x + c_By_x * d_Ez_x
            Bx = Bx - M * (d_Ez_y / kappa_y + Psi_Bx_y)
            By = By + M * (d_Ez_x / kappa_x + Psi_By_x)
            Hx = Bx / MRxx
            Hy = By / MRyy

            d_Hx_y, d_Hy_x = self._curl_h(Hx, Hy, dx, dy, per_x, per_y)

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

                    t_half = t_index * self.dt + self.dt / 2.0
                    kx = float(np.cos(s["angle"]))
                    ky = float(np.sin(s["angle"]))

                    if ix_lo < self.Nx:
                        t_edge = t_half - s["Hy_delay_xlo"]
                        Hy_src_xlo = -kx * torch.as_tensor(self._g(s, t_edge), dtype=self.torch_dtype, device=self.device)
                        for j_off, j in enumerate(range(iy_lo, iy_hi + 1)):
                            if 0 <= j < self.Ny:
                                d_Hy_x[ix_lo, j] -= Hy_src_xlo[j_off] / dx

                    if ix_hi < self.Nx:
                        t_edge = t_half - s["Hy_delay_xhi"]
                        Hy_src_xhi = -kx * torch.as_tensor(self._g(s, t_edge), dtype=self.torch_dtype, device=self.device)
                        idx = ix_hi + 1
                        if idx < self.Nx:
                            for j_off, j in enumerate(range(iy_lo, iy_hi + 1)):
                                if 0 <= j < self.Ny:
                                    d_Hy_x[idx, j] += Hy_src_xhi[j_off] / dx

                    if iy_lo < self.Ny:
                        t_edge = t_half - s["Hx_delay_ylo"]
                        Hx_src_ylo = ky * torch.as_tensor(self._g(s, t_edge), dtype=self.torch_dtype, device=self.device)
                        for i_off, i in enumerate(range(ix_lo, ix_hi + 1)):
                            if 0 <= i < self.Nx:
                                d_Hx_y[i, iy_lo] -= Hx_src_ylo[i_off] / dy

                    if iy_hi < self.Ny:
                        t_edge = t_half - s["Hx_delay_yhi"]
                        Hx_src_yhi = ky * torch.as_tensor(self._g(s, t_edge), dtype=self.torch_dtype, device=self.device)
                        idx = iy_hi + 1
                        if idx < self.Ny:
                            for i_off, i in enumerate(range(ix_lo, ix_hi + 1)):
                                if 0 <= i < self.Nx:
                                    d_Hx_y[i, idx] += Hx_src_yhi[i_off] / dy

                elif kind == 'waveguide-y':
                    n_eff = s["n_eff"]
                    delay = t_index * self.dt + self.dy * n_eff / (2 * self.c0) + self.dt / 2.0
                    H_src = -torch.as_tensor(self._g(s, delay), dtype=self.torch_dtype, device=self.device)
                    y = s["iy0"]
                    i0, i1 = s["ix0"], s["ix1"]
                    lo, hi = (min(i0, i1), max(i0, i1))
                    for i in range(lo, hi):
                        if 0 <= y < self.Ny:
                            d_Hx_y[i, y] -= (H_src / dy) * float(s["Hx_src"][i - lo])

                elif kind == 'waveguide-x':
                    n_eff = s["n_eff"]
                    delay = t_index * self.dt + self.dx * n_eff / (2 * self.c0) + self.dt / 2.0
                    H_src = -torch.as_tensor(self._g(s, delay), dtype=self.torch_dtype, device=self.device)
                    x = s["ix0"]
                    j0, j1 = s["iy0"], s["iy1"]
                    lo, hi = (min(j0, j1), max(j0, j1))
                    for j in range(lo, hi):
                        if 0 <= x < self.Nx:
                            d_Hy_x[x, j] += (H_src / dx) * float(s["Hy_src"][j - lo])

            Psi_Dz_x = b_Dz_x * Psi_Dz_x + c_Dz_x * d_Hy_x
            Psi_Dz_y = b_Dz_y * Psi_Dz_y + c_Dz_y * d_Hx_y
            Dz = Dz + M * (d_Hy_x / kappa_x - d_Hx_y / kappa_y + Psi_Dz_x - Psi_Dz_y)

            t_now = t_index * self.dt
            for s in self.sources:
                kind = s.get("kind")
                if kind == 'point':
                    i, j = s["ix0"], s["iy0"]
                    if 0 <= i < self.Nx and 0 <= j < self.Ny:
                        Dz[i, j] += torch.as_tensor(self._g(s, t_now), dtype=self.torch_dtype, device=self.device)
                elif kind == 'line-soft':
                    val = torch.as_tensor(self._g(s, t_now), dtype=self.torch_dtype, device=self.device)
                    if s["ix0"] != s["ix1"]:
                        y = s["iy0"]
                        lo = min(s["ix0"], s["ix1"])
                        hi = max(s["ix0"], s["ix1"])
                        if 0 <= y < self.Ny:
                            for i in range(lo, hi):
                                if 0 <= i < self.Nx:
                                    Dz[i, y] += val
                    else:
                        x = s["ix0"]
                        lo = min(s["iy0"], s["iy1"])
                        hi = max(s["iy0"], s["iy1"])
                        if 0 <= x < self.Nx:
                            for j in range(lo, hi):
                                if 0 <= j < self.Ny:
                                    Dz[x, j] += val

            Ez = Dz / ERzz

            if monitor_results:
                Hx_center = self._avg_with_neighbor_torch(0.5 * (Hx + Hx_prev), axis=1,
                                                          periodic=per_y, direction=-1)
                Hy_center = self._avg_with_neighbor_torch(0.5 * (Hy + Hy_prev), axis=0,
                                                          periodic=per_x, direction=-1)

            for buf in monitor_results:
                if buf["it0"] <= t_index < buf["it1"]:
                    k = t_index - buf["it0"]
                    if buf["orientation"] == "horizontal":
                        ez_slice = Ez[buf["_slx"], buf["_y"]]
                        hx_slice = Hx_center[buf["_slx"], buf["_y"]]
                        hy_slice = Hy_center[buf["_slx"], buf["_y"]]
                    else:
                        ez_slice = Ez[buf["_x"], buf["_sly"]]
                        hx_slice = Hx_center[buf["_x"], buf["_sly"]]
                        hy_slice = Hy_center[buf["_x"], buf["_sly"]]

                    buf["Ez"][k, :] = ez_slice.detach().cpu().numpy()
                    buf["Hx"][k, :] = hx_slice.detach().cpu().numpy()
                    buf["Hy"][k, :] = hy_slice.detach().cpu().numpy()

            if monitor_results:
                Hx_prev = Hx.clone()
                Hy_prev = Hy.clone()

            if self.is_include_history and (t_index % self.record_stride) == 0:
                self.Hx_history[rec_idx, :, :] = Hx.detach().cpu().numpy()
                self.Hy_history[rec_idx, :, :] = Hy.detach().cpu().numpy()
                self.Ez_history[rec_idx, :, :] = Ez.detach().cpu().numpy()
                self.Dz_history[rec_idx, :, :] = Dz.detach().cpu().numpy()
                rec_idx += 1

        self.monitor_results = []
        for buf in monitor_results:
            out = {k: v for k, v in buf.items() if not k.startswith("_")}
            self.monitor_results.append(out)

        self.Bx = Bx.detach().cpu().numpy()
        self.By = By.detach().cpu().numpy()
        self.Hx = Hx.detach().cpu().numpy()
        self.Hy = Hy.detach().cpu().numpy()
        self.Dz = Dz.detach().cpu().numpy()
        self.Ez = Ez.detach().cpu().numpy()

        self.Psi_Bx_y = Psi_Bx_y.detach().cpu().numpy()
        self.Psi_By_x = Psi_By_x.detach().cpu().numpy()
        self.Psi_Dz_x = Psi_Dz_x.detach().cpu().numpy()
        self.Psi_Dz_y = Psi_Dz_y.detach().cpu().numpy()

        self.d_Ez_y = d_Ez_y.detach().cpu().numpy()
        self.d_Ez_x = d_Ez_x.detach().cpu().numpy()
        self.d_Hx_y = d_Hx_y.detach().cpu().numpy()
        self.d_Hy_x = d_Hy_x.detach().cpu().numpy()
