from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

try:
    from . import _cython_kernel_1d as _cython_kernel
except (ImportError, ValueError):
    try:
        import _cython_kernel_1d as _cython_kernel
    except ImportError:
        _cython_kernel = None


class FDTD_1D:
    def __init__(self, z_range, Nz, f_max, Nt, dt=None, subpixel=16):
        if z_range <= 0 or Nz < 1 or f_max <= 0 or Nt < 1:
            raise ValueError("z_range, f_max, and Nt must be positive, and Nz must be at least 1.")
        if not isinstance(subpixel, (int, np.integer)) or subpixel < 1:
            raise ValueError("subpixel must be a positive integer.")

        self.eps0 = 8.85e-12
        self.mu0 = 4e-7 * np.pi
        self.c0 = 1 / np.sqrt(self.eps0 * self.mu0)

        self.z_range = float(z_range)
        self.Nz = int(Nz)
        self.dz = z_range / Nz
        self.subpixel = int(subpixel)
        self.z_Ey = np.arange(self.Nz + 1, dtype=float) * self.dz
        self.z_Hx = (np.arange(self.Nz, dtype=float) + 0.5) * self.dz

        # Material is assigned to cells first. Field/material arrays carrying a
        # component suffix live at that component's Yee location.
        self.ER = np.ones(Nz)
        self.MR = np.ones(Nz)
        self.ER_Ey = np.ones(Nz + 1)
        self.MR_Ey = np.ones(Nz + 1)
        self.ER_Hx = np.ones(Nz)
        self.MR_Hx = np.ones(Nz)
        self.mEy = np.ones(Nz + 1)
        self.mHx = np.ones(Nz)

        self.Ey = np.zeros(Nz + 1)
        self.Hx = np.zeros(Nz)

        # Perfect conductors are geometric/update constraints, not artificial
        # material values. A PEC cell constrains both surrounding Ey nodes;
        # a PMC cell constrains its cell-centred Hx sample.
        self.PEC_cells = np.zeros(Nz, dtype=bool)
        self.PMC_cells = np.zeros(Nz, dtype=bool)
        self.PEC_boundary_Ey = np.zeros(Nz + 1, dtype=bool)
        self.PMC_boundary_Hx = np.zeros(Nz, dtype=bool)
        self.PEC_Ey = np.zeros(Nz + 1, dtype=bool)
        self.PMC_Hx = np.zeros(Nz, dtype=bool)
        self.Ey_update_coeff = np.ones(Nz + 1)
        self.Hx_update_coeff = np.ones(Nz)

        self.Nt = Nt
        self.f_max = f_max
        dt_cfl = self.dz / (2 * self.c0)
        dt_freq_sampling = 1 / (20 * self.f_max)
        self.dt = dt if dt is not None else min(dt_cfl, dt_freq_sampling)

        self.src_index = None
        self.src_amplitude = 1.0
        self.src_t0 = None
        self.src_tw = None
        self.avg_freq = None

        self.left_absorbing_boundary = False
        self.right_absorbing_boundary = False
        self.left_boundary_material = None
        self.right_boundary_material = None

        self.ey_left_past = 0.0
        self.ey_right_past = 0.0

        self.Nf = min(100, Nt)
        self.REF = np.zeros(self.Nf, dtype=complex)
        self.TRN = np.zeros(self.Nf, dtype=complex)
        self.SRC = np.zeros(self.Nf, dtype=complex)
        self._cython_kernel = _cython_kernel
        self._use_cython_kernel = self._cython_kernel is not None

    @staticmethod
    def suggest_dx_dt(max_eps_r, max_mu_r, f_max, cells_per_wavelength=25, courant_factor=0.7,
                      time_samples_per_period=40):
        """
        Suggest a high-accuracy spatial step and time step from the worst-case material
        and highest simulated frequency.

        Returns a dictionary containing:
          - dz: recommended 1D cell size in meters
          - dx: alias of dz for API consistency with the 2D solvers
          - dt: recommended time step in seconds
          - lambda_min: shortest wavelength in the modeled material
          - refractive_index_max: sqrt(max_eps_r * max_mu_r)
        """
        if max_eps_r <= 0 or max_mu_r <= 0 or f_max <= 0:
            raise ValueError("max_eps_r, max_mu_r, and f_max must all be positive.")
        if cells_per_wavelength <= 0 or courant_factor <= 0 or time_samples_per_period <= 0:
            raise ValueError("cells_per_wavelength, courant_factor, and time_samples_per_period must be positive.")

        eps0 = 8.85e-12
        mu0 = 4e-7 * np.pi
        c0 = 1 / np.sqrt(eps0 * mu0)

        n_max = np.sqrt(max_eps_r * max_mu_r)
        lambda_min = c0 / (f_max * n_max)
        dz = lambda_min / cells_per_wavelength

        dt_cfl = dz / (2 * c0)
        dt_freq_sampling = 1.0 / (time_samples_per_period * f_max)
        dt = min(courant_factor * dt_cfl, dt_freq_sampling)

        return {
            "dz": dz,
            "dx": dz,
            "dt": dt,
            "lambda_min": lambda_min,
            "refractive_index_max": n_max,
            "cells_per_wavelength": cells_per_wavelength,
            "courant_factor": courant_factor,
            "time_samples_per_period": time_samples_per_period,
        }

    def _average_material_to_yee(self):
        """Map cell material onto the locations of the Yee components."""
        self.ER_Ey[[0, -1]] = self.ER[[0, -1]]
        self.MR_Ey[[0, -1]] = self.MR[[0, -1]]
        if self.Nz > 1:
            self.ER_Ey[1:-1] = 0.5 * (self.ER[:-1] + self.ER[1:])
            self.MR_Ey[1:-1] = 0.5 * (self.MR[:-1] + self.MR[1:])
        self.ER_Hx[:] = self.ER
        self.MR_Hx[:] = self.MR

    def _init_mEy_mHx(self):
        self._average_material_to_yee()
        self.mEy = self.Ey_update_coeff * self.c0 * self.dt / self.ER_Ey
        self.mHx = self.Hx_update_coeff * self.c0 * self.dt / self.MR_Hx

    @staticmethod
    def _parse_special_material(ER=None, MR=None, material=None):
        values = [value.upper() for value in (material, ER, MR) if isinstance(value, str)]
        if not values:
            return None
        if any(value not in {"PEC", "PMC"} for value in values):
            raise ValueError("Special material must be 'PEC' or 'PMC'.")
        if len(set(values)) != 1:
            raise ValueError("An object cannot be both PEC and PMC.")
        return values[0]

    def _refresh_conductor_masks(self):
        self.PEC_Ey[:] = self.PEC_boundary_Ey
        self.PEC_Ey[:-1] |= self.PEC_cells
        self.PEC_Ey[1:] |= self.PEC_cells
        self.PMC_Hx[:] = self.PMC_boundary_Hx | self.PMC_cells
        self.Ey_update_coeff = (~self.PEC_Ey).astype(float)
        self.Hx_update_coeff = (~self.PMC_Hx).astype(float)

    def _mark_special_cells(self, cells, material):
        if material == "PEC":
            self.PEC_cells[cells] = True
            self.PMC_cells[cells] = False
        else:
            self.PMC_cells[cells] = True
            self.PEC_cells[cells] = False
        self._refresh_conductor_masks()

    @staticmethod
    def _cython_compatible(*arrays):
        return all(array.dtype == np.float64 and array.flags.c_contiguous for array in arrays)

    def H_Update(self):
        used = self._cython_kernel is not None and self._cython_compatible(self.Hx, self.Ey, self.mHx)
        if used:
            self._cython_kernel.update_h(self.Hx, self.Ey, self.mHx, self.dz)
        if not used:
            for nz in range(self.Nz):
                self.Hx[nz] += self.mHx[nz] * (self.Ey[nz + 1] - self.Ey[nz]) / self.dz
        self.Hx[self.PMC_Hx] = 0.0

    def E_Update(self):
        used = self._cython_kernel is not None and self._cython_compatible(self.Ey, self.Hx, self.mEy)
        if used:
            self._cython_kernel.update_e(self.Ey, self.Hx, self.mEy, self.dz)
        if not used:
            for nz in range(1, self.Nz):
                self.Ey[nz] += self.mEy[nz] * (self.Hx[nz] - self.Hx[nz - 1]) / self.dz

        if not self.left_absorbing_boundary:
            self.Ey[0] += self.mEy[0] * self.Hx[0] / self.dz
        else:
            adjacent = self.Ey[1]
            S = self.c0 * self.dt / (self.dz * np.sqrt(self.ER[0] * self.MR[0]))
            self.Ey[0] = self.ey_left_past + (S - 1) / (S + 1) * (adjacent - self.Ey[0])
            self.ey_left_past = adjacent

        if not self.right_absorbing_boundary:
            self.Ey[-1] -= self.mEy[-1] * self.Hx[-1] / self.dz
        else:
            adjacent = self.Ey[-2]
            S = self.c0 * self.dt / (self.dz * np.sqrt(self.ER[-1] * self.MR[-1]))
            self.Ey[-1] = self.ey_right_past + (S - 1) / (S + 1) * (adjacent - self.Ey[-1])
            self.ey_right_past = adjacent

        self.Ey[self.PEC_Ey] = 0.0

    def add_object(self, ER=None, MR=None, region=None, subpixel=None, material=None):
        """
        Assign material to cells before it is averaged onto the Yee grid.

        region: either
          - Python slice of whole-cell indices, e.g. ``slice(30, 40)``.
          - A tuple/list of absolute positions in metres, ``(z_start, z_end)``.
            Boundary cells are volume-averaged using ``subpixel`` midpoint
            samples (the solver default is 16).
        """
        special = self._parse_special_material(ER, MR, material)
        if special is None:
            if ER is None or MR is None:
                raise ValueError("ER and MR are required for a dielectric/magnetic object.")
            ER = float(ER)
            MR = float(MR)
            if ER <= 0 or MR <= 0:
                raise ValueError("ER and MR must be positive.")

        if isinstance(region, slice):
            if special is not None:
                cells = np.zeros(self.Nz, dtype=bool)
                cells[region] = True
                self._mark_special_cells(cells, special)
                return
            else:
                self.ER[region] = ER
                self.MR[region] = MR
        elif isinstance(region, (tuple, list)) and len(region) == 2:
            z0, z1 = float(region[0]), float(region[1])
            if z1 < z0:
                z0, z1 = z1, z0
            z0 = float(np.clip(z0, 0.0, self.z_range))
            z1 = float(np.clip(z1, 0.0, self.z_range))
            if special is not None:
                cells = (self.z_Hx >= z0) & (self.z_Hx < z1)
                self._mark_special_cells(cells, special)
                return
            nsub = self.subpixel if subpixel is None else subpixel
            if not isinstance(nsub, (int, np.integer)) or nsub < 1:
                raise ValueError("subpixel must be a positive integer.")

            offsets = (np.arange(int(nsub), dtype=float) + 0.5) / int(nsub)
            sample_z = (np.arange(self.Nz, dtype=float)[:, None] + offsets[None, :]) * self.dz
            fill = np.mean((sample_z >= z0) & (sample_z < z1), axis=1)
            touched = fill > 0.0
            self.ER[touched] = ((1.0 - fill[touched]) * self.ER[touched]
                                + fill[touched] * ER)
            self.MR[touched] = ((1.0 - fill[touched]) * self.MR[touched]
                                + fill[touched] * MR)
        else:
            raise TypeError("region must be a slice or a (z_start, z_end) tuple in meters.")

        # Keep component material inspectable immediately after geometry edits.
        self._average_material_to_yee()
    def set_boundary(self, left: str = "absorbing", right: str = "absorbing"):
        left = str(left).lower()
        right = str(right).lower()
        valid = {"absorbing", "a", "electric", "e", "pec", "magnetic", "m", "pmc"}
        if left not in valid or right not in valid:
            raise ValueError("Boundary must be absorbing, electric/PEC, or magnetic/PMC.")

        self.PEC_boundary_Ey[[0, -1]] = False
        self.PMC_boundary_Hx[[0, -1]] = False

        if left in {"electric", "e", "pec"}:
            self.PEC_boundary_Ey[0] = True
            self.left_absorbing_boundary = False
            self.left_boundary_material = "PEC"
        elif left in {"magnetic", "m", "pmc"}:
            self.PMC_boundary_Hx[0] = True
            self.left_absorbing_boundary = False
            self.left_boundary_material = "PMC"

        else:
            self.left_absorbing_boundary = True
            self.left_boundary_material = None
        if right in {"electric", "e", "pec"}:
            self.PEC_boundary_Ey[-1] = True
            self.right_absorbing_boundary = False
            self.right_boundary_material = "PEC"
        elif right in {"magnetic", "m", "pmc"}:
            self.PMC_boundary_Hx[-1] = True
            self.right_absorbing_boundary = False
            self.right_boundary_material = "PMC"
        else:
            self.right_absorbing_boundary = True
            self.right_boundary_material = None

        self._refresh_conductor_masks()
    def add_source(self, src_position, amplitude=1.0, t0=None, tw=None, is_show=True):
        if isinstance(src_position, (int, np.integer)):
            self.src_index = int(src_position)
        elif isinstance(src_position, (float, np.floating)) and 0 <= src_position <= self.z_range:
            self.src_index = int(np.round(src_position / self.dz))
        else:
            raise TypeError(
                "The source position must be an integer for the index or a float for the absolute position."
                "It must also be within the simulation range.")
        if not 1 <= self.src_index < self.Nz:
            raise ValueError("The source must be on an interior Ey location (index 1 through Nz - 1).")
        self.src_amplitude = amplitude
        self.src_tw = tw if tw is not None else 0.5 / self.f_max
        self.src_t0 = t0 if t0 is not None else 4 * self.src_tw

        # Precompute average frequency (spectral centroid) for later PML use
        t = np.arange(0, self.Nt * self.dt, self.dt)
        pulse = self.src_amplitude * np.exp(-((t - self.src_t0) / self.src_tw) ** 2)
        if not np.isclose(pulse[0], 0, atol=1e-04):
            raise ValueError("Warning: The source is non-zero at the start.")
        freq = np.fft.fftfreq(len(t), d=self.dt)
        spectrum = np.fft.fft(pulse)
        pos_mask = freq >= 0
        freq_pos = freq[pos_mask]
        spec_pos = np.abs(spectrum[pos_mask]) + 1e-30  # avoid divide-by-zero
        self.avg_freq = float(np.sum(freq_pos * spec_pos) / np.sum(spec_pos))

        if is_show:
            # Plot time domain and frequency domain
            fig, axs = plt.subplots(2, 1, figsize=(6, 6))
            axs[0].plot(t * 1e9, pulse)
            axs[0].set_xlabel('Time (ns)')
            axs[0].set_ylabel('Amplitude')
            axs[0].set_title('Source (Time Domain)')

            axs[1].plot(freq_pos / 1e9, spec_pos)
            axs[1].set_xlim(0, 2 * self.f_max / 1e9)
            axs[1].set_xlabel('Frequency (GHz)')
            axs[1].set_ylabel('Magnitude')
            axs[1].set_title(f'Source Spectrum (Avg f ≈ {self.avg_freq / 1e9:.3f} GHz)')

            plt.tight_layout()
            plt.show()

    def _pulse(self, t) -> float | Any:
        return self.src_amplitude * np.exp(-((t - self.src_t0) / self.src_tw) ** 2)

    def run(self):
        if self.src_index is None:
            raise RuntimeError("Call add_source() before run().")
        self._init_mEy_mHx()
        Hx_history = np.zeros((self.Nt, self.Nz), dtype=complex)
        Ey_history = np.zeros((self.Nt, self.Nz + 1), dtype=complex)

        REF_history = np.zeros((self.Nt, self.Nf), dtype=complex)
        TRN_history = np.zeros((self.Nt, self.Nf), dtype=complex)
        SRC_history = np.zeros((self.Nt, self.Nf), dtype=complex)

        fn = np.linspace(0, self.f_max, self.Nf)
        Kn = np.exp(-1j * 2 * np.pi * fn * self.dt)

        C_REF = np.sqrt(np.sqrt(self.MR_Ey[1] / self.ER_Ey[1]))
        C_TRN = np.sqrt(np.sqrt(self.MR_Ey[-2] / self.ER_Ey[-2]))
        C_SRC = np.sqrt(np.sqrt(self.MR_Ey[self.src_index] / self.ER_Ey[self.src_index]))

        # Add tqdm progress bar around the loop
        for t_index in tqdm(range(self.Nt), desc="Running simulation", unit="step"):
            self.H_Update()

            if self.src_index is not None:
                Ey_src = self._pulse(t_index * self.dt)
                self.Hx[self.src_index - 1] -= self.mHx[self.src_index - 1] / self.dz * Ey_src

            self.E_Update()

            if self.src_index is not None:
                EtaR_inv = np.sqrt(self.ER_Ey[self.src_index] / self.MR_Ey[self.src_index])
                n = np.sqrt(self.ER_Ey[self.src_index] * self.MR_Ey[self.src_index])
                Hx_src = -EtaR_inv * self._pulse(
                    t_index * self.dt + n * self.dz / (2 * self.c0) + self.dt / 2
                )
                self.Ey[self.src_index] -= self.mEy[self.src_index] / self.dz * Hx_src

            for nf in range(self.Nf):
                self.REF[nf] += self.dt * Kn[nf] ** t_index * self.Ey[1] / C_REF
                self.TRN[nf] += self.dt * Kn[nf] ** t_index * self.Ey[-2] / C_TRN
                self.SRC[nf] += self.dt * Kn[nf] ** t_index * self._pulse(t_index * self.dt) / C_SRC

            Hx_history[t_index, :] = self.Hx.copy()
            Ey_history[t_index, :] = self.Ey.copy()
            REF_history[t_index, :] = self.REF.copy()
            TRN_history[t_index, :] = self.TRN.copy()
            SRC_history[t_index, :] = self.SRC.copy()

        self.Hx_history = Hx_history
        self.Ey_history = Ey_history
        self.REF_history = REF_history
        self.TRN_history = TRN_history
        self.SRC_history = SRC_history

    @staticmethod
    def _cell_intervals(mask):
        """Return half-open index intervals for contiguous True cells."""
        padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
        changes = np.diff(padded)
        return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))

    def _draw_conductor_regions(self, ax, position_scale=1.0, add_labels=True):
        """Draw PEC in dashed yellow and PMC in dashed blue."""
        artists = []
        labeled = {"PEC": False, "PMC": False}
        for name, mask, color in (("PEC", self.PEC_cells, "yellow"),
                                  ("PMC", self.PMC_cells, "blue")):
            for start, stop in self._cell_intervals(mask):
                label = name if add_labels and not labeled[name] else None
                artists.append(ax.axvspan(start * self.dz * position_scale,
                                          stop * self.dz * position_scale,
                                          facecolor="none", edgecolor=color,
                                          linestyle="--", linewidth=1.8,
                                          label=label, zorder=6))
                labeled[name] = True

        for position, name in ((0.0, self.left_boundary_material),
                               (self.z_range, self.right_boundary_material)):
            if name in labeled:
                color = "yellow" if name == "PEC" else "blue"
                label = name if add_labels and not labeled[name] else None
                artists.append(ax.axvline(position * position_scale, color=color,
                                          linestyle="--", linewidth=1.8,
                                          label=label, zorder=6))
                labeled[name] = True
        return artists

    def show_animation(self, fps=1):
        if fps > 1:
            print(
                "Warning: The FPS may be too high to render a smooth animation.\n"
                "This Python code is not responsible for any laptops or PCs that explode.")
        x_E = self.z_Ey
        x_H = self.z_Hx

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 6), gridspec_kw={'height_ratios': [3, 1, 2]})

        # Top: fields
        line_E, = ax1.plot(x_E * 1e3, self.Ey_history[0].real, label='E field', color='red')
        line_H, = ax1.plot(x_H * 1e3, self.Hx_history[0].real, label='H field', color='blue')
        ax1.axvline(x=self.src_index * self.dz * 1e3, color='green', linestyle='--')
        self._draw_conductor_regions(ax1, position_scale=1e3, add_labels=True)
        ax1.set_title('1D FDTD Simulation')
        ax1.set_xlim(0, self.z_range * 1e3)
        ax1.set_ylim(-3, 3)
        ax1.set_ylabel('Amplitude')
        ax1.legend()

        # Add annotation for time
        time_text = ax1.text(0.02, 0.8, '', transform=ax1.transAxes,
                             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # Middle: material profiles
        ax2.step(x_H * 1e3, self.ER.real, where='mid', label='εr cells (real)')
        ax2.step(x_H * 1e3, self.MR.real, where='mid', label='μr cells (real)')
        self._draw_conductor_regions(ax2, position_scale=1e3, add_labels=True)
        ax2.set_xlim(0, self.z_range * 1e3)
        ax2.set_xlabel('Position (mm)')
        ax2.set_ylabel('Value')
        ax2.legend(loc='upper right')
        ax2.set_title('Material Profiles')

        # Bottom: transmission and reflection
        fn = np.linspace(0, self.f_max, self.Nf)
        line_REF, = ax3.plot(fn / 1e9, abs(self.REF_history[0] / self.SRC_history[0]) ** 2, label='Reflection',
                             color='red')
        line_TRN, = ax3.plot(fn / 1e9, abs(self.TRN_history[0] / self.SRC_history[0]) ** 2, label='Transmission',
                             color='blue')
        ax3.set_xlim(0, self.f_max / 1e9)
        ax3.set_ylim(-0.1, 1.1)
        ax3.set_xlabel('Frequency (GHz)')
        ax3.set_ylabel('Magnitude')
        ax3.legend(loc='upper right')
        ax3.set_title('Transmission and Reflection')
        ax3.grid()

        def update(frame):
            # update fields
            line_E.set_ydata(self.Ey_history[frame].real)
            line_H.set_ydata(self.Hx_history[frame].real)

            REF_data = abs(self.REF_history[frame] / self.SRC_history[frame]) ** 2
            line_REF.set_ydata(REF_data)
            TRN_data = abs(self.TRN_history[frame] / self.SRC_history[frame]) ** 2
            line_TRN.set_ydata(TRN_data)

            # compute times
            t_E = frame * self.dt
            t_H = (frame + 0.5) * self.dt

            # update annotation text
            time_text.set_text(f"t(E) = {t_E * 1e12:.5f} ps\n"
                               f"t(H) = {t_H * 1e12:.5f} ps")

            return line_E, line_H, time_text, line_REF, line_TRN

        ani = FuncAnimation(fig, update, frames=self.Nt, interval=1 / fps, blit=True, repeat=False)
        plt.tight_layout()
        plt.show()
