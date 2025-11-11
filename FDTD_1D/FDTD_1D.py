from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm


class FDTD_1D:
    def __init__(self, z_range, Nz, f_max, Nt, dt=None):
        self.eps0 = 8.85e-12
        self.mu0 = 4e-7 * np.pi
        self.c0 = 1 / np.sqrt(self.eps0 * self.mu0)

        self.z_range = z_range
        self.Nz = Nz
        self.dz = z_range / Nz

        self.ER = np.ones(Nz)
        self.MR = np.ones(Nz)
        self.mEy = np.ones(Nz)
        self.mHx = np.ones(Nz)

        self.Ey = np.zeros(Nz)
        self.Hx = np.zeros(Nz)

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

        self.ey_past = 0
        self.hx_past = 0

        self.Nf = min(100, Nt)
        self.REF = np.zeros(self.Nf, dtype=complex)
        self.TRN = np.zeros(self.Nf, dtype=complex)
        self.SRC = np.zeros(self.Nf, dtype=complex)

    def _init_mEy_mHx(self):
        self.mEy = self.c0 * self.dt / self.ER
        self.mHx = self.c0 * self.dt / self.MR

    def H_Update(self):
        for nz in range(0, self.Nz - 1):
            self.Hx[nz] += self.mHx[nz] * (self.Ey[nz + 1] - self.Ey[nz]) / self.dz

        if self.right_absorbing_boundary is not True:
            self.Hx[self.Nz - 1] += self.mHx[self.Nz - 1] * (-self.Ey[self.Nz - 1]) / self.dz
        else:
            S = self.c0 * self.dt / (self.dz * np.sqrt(self.ER[self.Nz - 1] * self.MR[self.Nz - 1]))
            self.Hx[self.Nz - 1] = self.hx_past + (S - 1) / (S + 1) * (self.Hx[self.Nz - 2] - self.Hx[self.Nz - 1])
            self.hx_past = self.Hx[self.Nz - 2]

    def E_Update(self):
        for nz in range(1, self.Nz):
            self.Ey[nz] += self.mEy[nz] * (self.Hx[nz] - self.Hx[nz - 1]) / self.dz

        if self.left_absorbing_boundary is not True:
            self.Ey[0] += self.mEy[0] * self.Hx[0] / self.dz
        else:
            S = self.c0 * self.dt / (self.dz * np.sqrt(self.ER[0] * self.MR[0]))
            self.Ey[0] = self.ey_past + (S - 1) / (S + 1) * (self.Ey[1] - self.Ey[0])
            self.ey_past = self.Ey[1]

    def _indices_from_z(self, z_start, z_end):
        i0 = int(np.clip(np.round(z_start / self.dz), 0, self.Nz - 1))
        i1 = int(np.clip(np.round(z_end / self.dz), 0, self.Nz))
        if i1 < i0:
            i0, i1 = i1, i0
        return slice(i0, i1)

    def add_object(self, ER, MR, region):
        """
        region: either
          - Python slice of indices (existing behavior), e.g., slice(30, 40)
          - Tuple/list of absolute positions in meters, (z_start, z_end)
        """
        if isinstance(region, slice):
            sl = region
        elif isinstance(region, (tuple, list)) and len(region) == 2:
            z0, z1 = float(region[0]), float(region[1])
            sl = self._indices_from_z(z0, z1)
        else:
            raise TypeError("region must be a slice or a (z_start, z_end) tuple in meters.")
        self.ER[sl] = ER
        self.MR[sl] = MR

    def set_boundary(self, left: str = "absorbing", right: str = "absorbing"):
        valid = {"absorbing", "a", "electric", "e", "magnetic", "m"}
        if left not in valid or right not in valid:
            raise ValueError("Boundary must be one of {'absorbing' / 'a', 'electric' / 'e' ,'magnetic' / 'm'}")

        # Apply electric/magnetic extremes to boundary cells
        if left == "electric" or left == "e":
            self.ER[0] = 1e8
            self.left_absorbing_boundary = False
        elif left == "magnetic" or left == "m":
            self.MR[0] = 1e8
            self.left_absorbing_boundary = False

        else:
            self.left_absorbing_boundary = True

        if right == "electric" or right == "e":
            self.ER[-1] = 1e8
            self.right_absorbing_boundary = False
        elif right == "magnetic" or right == "m":
            self.MR[-1] = 1e8
            self.right_absorbing_boundary = False
        else:
            self.right_absorbing_boundary = True

    def add_source(self, src_position, amplitude=1.0, t0=None, tw=None, is_show=True):
        if isinstance(src_position, int):
            self.src_index = src_position
        elif isinstance(src_position, float) and 0 <= src_position <= self.z_range:
            self.src_index = int(np.round(src_position / self.dz))
        else:
            raise TypeError(
                "The source position must be an integer for the index or a float for the absolute position."
                "It must also be within the simulation range.")
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
        self._init_mEy_mHx()
        Hx_history = np.zeros((self.Nt, self.Nz), dtype=complex)
        Ey_history = np.zeros((self.Nt, self.Nz), dtype=complex)

        REF_history = np.zeros((self.Nt, self.Nf), dtype=complex)
        TRN_history = np.zeros((self.Nt, self.Nf), dtype=complex)
        SRC_history = np.zeros((self.Nt, self.Nf), dtype=complex)

        fn = np.linspace(0, self.f_max, self.Nf)
        Kn = np.exp(-1j * 2 * np.pi * fn * self.dt)

        C_REF = np.sqrt(np.sqrt(self.MR[1] / self.ER[1]))
        C_TRN = np.sqrt(np.sqrt(self.MR[-2] / self.ER[-2]))
        C_SRC = np.sqrt(np.sqrt(self.MR[self.src_index] / self.ER[self.src_index]))

        # Add tqdm progress bar around the loop
        for t_index in tqdm(range(self.Nt), desc="Running simulation", unit="step"):
            self.H_Update()

            if self.src_index is not None:
                Ey_src = self._pulse(t_index * self.dt)
                self.Hx[self.src_index - 1] -= self.mHx[self.src_index - 1] / self.dz * Ey_src

            self.E_Update()

            if self.src_index is not None:
                EtaR_inv = np.sqrt(self.ER[self.src_index] / self.MR[self.src_index])
                n = np.sqrt(self.ER[self.src_index] * self.MR[self.src_index])
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

    def show_animation(self, fps=1):
        if fps > 1:
            print(
                "Warning: The FPS may be too high to render a smooth animation.\n"
                "This Python code is not responsible for any laptops or PCs that explode.")
        x_E = np.linspace(0, self.z_range, self.Nz)
        x_H = x_E + self.dz / 2

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 6), gridspec_kw={'height_ratios': [3, 1, 2]})

        # Top: fields
        line_E, = ax1.plot(x_E * 1e3, self.Ey_history[0].real, label='E field', color='red')
        line_H, = ax1.plot(x_H * 1e3, self.Hx_history[0].real, label='H field', color='blue')
        ax1.axvline(x=self.src_index * self.dz * 1e3, color='green', linestyle='--')
        ax1.set_title('1D FDTD Simulation')
        ax1.set_xlim(0, self.z_range * 1e3)
        ax1.set_ylim(-3, 3)
        ax1.set_ylabel('Amplitude')
        ax1.legend()

        # Add annotation for time
        time_text = ax1.text(0.02, 0.8, '', transform=ax1.transAxes,
                             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # Middle: material profiles
        ax2.plot(x_E * 1e3, self.ER.real, label='εr (real)')
        ax2.plot(x_E * 1e3, self.MR.real, label='μr (real)')
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
