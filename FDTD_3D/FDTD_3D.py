import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


class FDTD_3D:
    """
    3D FDTD skeleton with CPML and multiple source types.

    Conventions mirror FDTD_2D_Ez:
      • normalized units with E, H on Cartesian grid
      • CPML via kappa/sigma/alpha and auxiliary Psi terms
      • source list supports point/line/plane/TF-SF/waveguide mode
      • visualization utilities: 3D vector plot or 2D slice
      • NF2FF (near-field to far-field) placeholder
    """

    def __init__(self, x_range, y_range, z_range, Nx, Ny, Nz, f_max, Nt, f_min=None, dt=None):
        # constants
        self.eps0 = 8.85e-12
        self.mu0 = 4e-7 * np.pi
        self.c0 = 1 / np.sqrt(self.eps0 * self.mu0)
        self.eta0 = np.sqrt(self.mu0 / self.eps0)

        # grid
        self.x_range = float(x_range)
        self.y_range = float(y_range)
        self.z_range = float(z_range)
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.Nz = int(Nz)
        self.dx = self.x_range / self.Nx
        self.dy = self.y_range / self.Ny
        self.dz = self.z_range / self.Nz

        # time
        self.Nt = int(Nt)
        self.f_min = f_min if f_min is not None else None
        self.f_max = float(f_max)
        dt_cfl = 1.0 / (self.c0 * np.sqrt((1 / self.dx ** 2) + (1 / self.dy ** 2) + (1 / self.dz ** 2)))
        dt_freq_sampling = 1.0 / (20 * self.f_max)
        self.dt = float(dt) if dt is not None else min(dt_cfl, dt_freq_sampling)

        # materials (relative)
        self.ERxx = np.ones((self.Nx, self.Ny, self.Nz))
        self.ERyy = np.ones((self.Nx, self.Ny, self.Nz))
        self.ERzz = np.ones((self.Nx, self.Ny, self.Nz))
        self.MRxx = np.ones((self.Nx, self.Ny, self.Nz))
        self.MRyy = np.ones((self.Nx, self.Ny, self.Nz))
        self.MRzz = np.ones((self.Nx, self.Ny, self.Nz))

        # fields (simple collocated storage; you can swap to Yee staggering later)
        self.Ex = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Ey = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Ez = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Hx = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Hy = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Hz = np.zeros((self.Nx, self.Ny, self.Nz))

        # CPML
        self.pml_width = None
        self.pml_order = 3
        self.pml_direction = 'xyz'
        self.sigma_max = None

        self.kappa_x = np.ones((self.Nx, self.Ny, self.Nz))
        self.kappa_y = np.ones((self.Nx, self.Ny, self.Nz))
        self.kappa_z = np.ones((self.Nx, self.Ny, self.Nz))
        self.alpha_x = np.zeros((self.Nx, self.Ny, self.Nz))
        self.alpha_y = np.zeros((self.Nx, self.Ny, self.Nz))
        self.alpha_z = np.zeros((self.Nx, self.Ny, self.Nz))
        self.sigma_x = np.zeros((self.Nx, self.Ny, self.Nz))
        self.sigma_y = np.zeros((self.Nx, self.Ny, self.Nz))
        self.sigma_z = np.zeros((self.Nx, self.Ny, self.Nz))

        # CPML coeffs (per-derivative)
        self.b_x = np.ones((self.Nx, self.Ny, self.Nz))
        self.b_y = np.ones((self.Nx, self.Ny, self.Nz))
        self.b_z = np.ones((self.Nx, self.Ny, self.Nz))
        self.c_x = np.zeros((self.Nx, self.Ny, self.Nz))
        self.c_y = np.zeros((self.Nx, self.Ny, self.Nz))
        self.c_z = np.zeros((self.Nx, self.Ny, self.Nz))

        # CPML auxiliary fields (Psi for each derivative component)
        self.Psi_Ex_y = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Psi_Ex_z = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Psi_Ey_x = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Psi_Ey_z = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Psi_Ez_x = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Psi_Ez_y = np.zeros((self.Nx, self.Ny, self.Nz))

        self.Psi_Hx_y = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Psi_Hx_z = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Psi_Hy_x = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Psi_Hy_z = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Psi_Hz_x = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Psi_Hz_y = np.zeros((self.Nx, self.Ny, self.Nz))

        # sources
        self.sources = []
        self.avg_freqs = []

    # ---------- PML ----------
    def add_PML(self, pml_width, order=3, direction='xyz', sigma_max=None,
                kappa_max=7, alpha_max=0.025, R0=1e-8):
        self.pml_width = int(pml_width) if isinstance(pml_width, int) else float(pml_width)
        self.pml_order = int(order)
        self.pml_direction = str(direction)

        if isinstance(pml_width, int):
            npx = npy = npz = max(1, int(pml_width))
        elif isinstance(pml_width, float):
            npx = max(1, int(np.ceil(pml_width / self.dx)))
            npy = max(1, int(np.ceil(pml_width / self.dy)))
            npz = max(1, int(np.ceil(pml_width / self.dz)))
        else:
            raise TypeError("pml_width must be int (cells) or float (meters).")

        Lx = npx * self.dx if npx > 0 else np.inf
        Ly = npy * self.dy if npy > 0 else np.inf
        Lz = npz * self.dz if npz > 0 else np.inf
        L = min(Lx, Ly, Lz)

        if sigma_max is None:
            self.sigma_max = -(self.pml_order + 1) * np.log10(R0) / (2 * self.eta0 * L)
        else:
            self.sigma_max = float(sigma_max)

        if 'x' in direction:
            for i in range(npx):
                s = self.sigma_max * ((npx - i) / npx) ** order
                k = 1 + (kappa_max - 1) * ((npx - i) / npx) ** order
                a = alpha_max * (i / npx) ** order
                self.sigma_x[i, :, :] = s
                self.sigma_x[-i - 1, :, :] = s
                self.kappa_x[i, :, :] = k
                self.kappa_x[-i - 1, :, :] = k
                self.alpha_x[i, :, :] = a
                self.alpha_x[-i - 1, :, :] = a

        if 'y' in direction:
            for j in range(npy):
                s = self.sigma_max * ((npy - j) / npy) ** order
                k = 1 + (kappa_max - 1) * ((npy - j) / npy) ** order
                a = alpha_max * (j / npy) ** order
                self.sigma_y[:, j, :] = s
                self.sigma_y[:, -j - 1, :] = s
                self.kappa_y[:, j, :] = k
                self.kappa_y[:, -j - 1, :] = k
                self.alpha_y[:, j, :] = a
                self.alpha_y[:, -j - 1, :] = a

        if 'z' in direction:
            for k in range(npz):
                s = self.sigma_max * ((npz - k) / npz) ** order
                kappa = 1 + (kappa_max - 1) * ((npz - k) / npz) ** order
                a = alpha_max * (k / npz) ** order
                self.sigma_z[:, :, k] = s
                self.sigma_z[:, :, -k - 1] = s
                self.kappa_z[:, :, k] = kappa
                self.kappa_z[:, :, -k - 1] = kappa
                self.alpha_z[:, :, k] = a
                self.alpha_z[:, :, -k - 1] = a

        if not ('x' in direction or 'y' in direction or 'z' in direction):
            raise TypeError("direction must contain 'x', 'y', or 'z'.")

        self._init_coeff()

    def _init_coeff(self):
        # b/c coeffs for CPML (same for E and H in this skeleton)
        ex = self.sigma_x / (self.eps0 * self.kappa_x) + self.alpha_x / self.eps0
        ey = self.sigma_y / (self.eps0 * self.kappa_y) + self.alpha_y / self.eps0
        ez = self.sigma_z / (self.eps0 * self.kappa_z) + self.alpha_z / self.eps0

        self.b_x = np.exp(-ex * self.dt)
        self.b_y = np.exp(-ey * self.dt)
        self.b_z = np.exp(-ez * self.dt)

        den_x = self.sigma_x * self.kappa_x + self.alpha_x * self.kappa_x ** 2
        den_y = self.sigma_y * self.kappa_y + self.alpha_y * self.kappa_y ** 2
        den_z = self.sigma_z * self.kappa_z + self.alpha_z * self.kappa_z ** 2

        self.c_x = np.zeros_like(self.sigma_x)
        self.c_y = np.zeros_like(self.sigma_y)
        self.c_z = np.zeros_like(self.sigma_z)

        good_x = den_x != 0
        good_y = den_y != 0
        good_z = den_z != 0
        self.c_x[good_x] = (self.sigma_x[good_x] / den_x[good_x]) * (self.b_x[good_x] - 1.0)
        self.c_y[good_y] = (self.sigma_y[good_y] / den_y[good_y]) * (self.b_y[good_y] - 1.0)
        self.c_z[good_z] = (self.sigma_z[good_z] / den_z[good_z]) * (self.b_z[good_z] - 1.0)

    # ---------- sources ----------
    def add_source(self, kind, ix0, iy0, iz0, ix1=None, iy1=None, iz1=None,
                   amplitude=1.0, t0=0.0, tw=1.0, f_min=None, f_max=None,
                   polarization='z', **kwargs):
        """
        kind: 'point' | 'line' | 'plane' | 'tfsf' | 'waveguide'
        polarization: 'x' | 'y' | 'z'
        ix1/iy1/iz1 define extents (half-open) for line/plane.
        """
        src = {
            "kind": str(kind),
            "ix0": int(ix0),
            "iy0": int(iy0),
            "iz0": int(iz0),
            "ix1": int(ix1) if ix1 is not None else None,
            "iy1": int(iy1) if iy1 is not None else None,
            "iz1": int(iz1) if iz1 is not None else None,
            "amplitude": float(amplitude),
            "t0": float(t0),
            "tw": float(tw),
            "f_min": f_min if f_min is not None else None,
            "f_max": f_max if f_max is not None else None,
            "pol": str(polarization).lower(),
        }
        src.update(kwargs)
        self.sources.append(src)

        # record spectral centroid for diagnostics
        t = np.arange(self.Nt) * self.dt
        wf = self._g(src, t)
        self.avg_freqs.append(self._spec_centroid(wf))

    def _g(self, s, t):
        amp = s["amplitude"]
        t0 = s["t0"]
        tw = s["tw"]
        fmin = s["f_min"]
        fmax = s["f_max"]

        t = np.asarray(t, dtype=float)

        if fmin is None:
            return amp * np.exp(-((t - t0) / tw) ** 2)

        if np.isclose(fmin, fmax):
            f0 = float(fmax)
            Tr = max(1 / max(f0, 1e-30), 1 * self.dt)
            tau = np.maximum(t - t0, 0.0)
            ramp = 1.0 - np.exp(-(tau / Tr) ** 3)
            return amp * ramp * np.sin(2 * np.pi * f0 * (t - t0))

        f0 = 0.5 * (fmin + fmax)
        return amp * np.sin(2 * np.pi * f0 * (t - t0)) * np.exp(-((t - t0) / tw) ** 2)

    def _spec_centroid(self, waveform):
        freq = np.fft.fftfreq(len(waveform), d=self.dt)
        S = np.fft.fft(waveform)
        pos = freq >= 0
        f = freq[pos]
        mag = np.abs(S[pos]) + 1e-30
        return float(np.sum(f * mag) / np.sum(mag))

    def apply_sources(self, n):
        t = n * self.dt
        for s in self.sources:
            val = self._g(s, t)
            pol = s["pol"]
            if s["kind"] == "point":
                self._apply_point_source(s, val, pol)
            elif s["kind"] == "line":
                self._apply_line_source(s, val, pol)
            elif s["kind"] == "plane":
                self._apply_plane_source(s, val, pol)
            elif s["kind"] == "tfsf":
                self._apply_tfsf_source(s, val, pol)
            elif s["kind"] == "waveguide":
                self._apply_waveguide_source(s, val, pol)

    def _apply_point_source(self, s, val, pol):
        ix, iy, iz = s["ix0"], s["iy0"], s["iz0"]
        if pol == 'x':
            self.Ex[ix, iy, iz] += val
        elif pol == 'y':
            self.Ey[ix, iy, iz] += val
        else:
            self.Ez[ix, iy, iz] += val

    def _apply_line_source(self, s, val, pol):
        ix0, iy0, iz0 = s["ix0"], s["iy0"], s["iz0"]
        ix1 = s["ix1"] if s["ix1"] is not None else ix0 + 1
        iy1 = s["iy1"] if s["iy1"] is not None else iy0 + 1
        iz1 = s["iz1"] if s["iz1"] is not None else iz0 + 1
        sl = (slice(ix0, ix1), slice(iy0, iy1), slice(iz0, iz1))
        if pol == 'x':
            self.Ex[sl] += val
        elif pol == 'y':
            self.Ey[sl] += val
        else:
            self.Ez[sl] += val

    def _apply_plane_source(self, s, val, pol):
        # plane source over rectangular slab
        self._apply_line_source(s, val, pol)

    def _apply_tfsf_source(self, s, val, pol):
        # TODO: implement full TF/SF boundary injection
        # placeholder: apply to plane region
        self._apply_plane_source(s, val, pol)

    def _apply_waveguide_source(self, s, val, pol):
        # TODO: compute modal distribution and inject; for now uniform over port
        self._apply_plane_source(s, val, pol)

    # ---------- update ----------
    def step(self, n):
        self._update_H()
        self._update_E()
        self.apply_sources(n)

    def _update_H(self):
        # curl E
        dEy_dz = (np.roll(self.Ey, -1, axis=2) - self.Ey) / self.dz
        dEz_dy = (np.roll(self.Ez, -1, axis=1) - self.Ez) / self.dy
        dEz_dx = (np.roll(self.Ez, -1, axis=0) - self.Ez) / self.dx
        dEx_dz = (np.roll(self.Ex, -1, axis=2) - self.Ex) / self.dz
        dEx_dy = (np.roll(self.Ex, -1, axis=1) - self.Ex) / self.dy
        dEy_dx = (np.roll(self.Ey, -1, axis=0) - self.Ey) / self.dx

        # CPML auxiliaries for H updates
        self.Psi_Hx_y = self.b_y * self.Psi_Hx_y + self.c_y * dEz_dy
        self.Psi_Hx_z = self.b_z * self.Psi_Hx_z + self.c_z * dEy_dz
        self.Psi_Hy_x = self.b_x * self.Psi_Hy_x + self.c_x * dEz_dx
        self.Psi_Hy_z = self.b_z * self.Psi_Hy_z + self.c_z * dEx_dz
        self.Psi_Hz_x = self.b_x * self.Psi_Hz_x + self.c_x * dEy_dx
        self.Psi_Hz_y = self.b_y * self.Psi_Hz_y + self.c_y * dEx_dy

        curlEx = (dEz_dy + self.Psi_Hx_y) - (dEy_dz + self.Psi_Hx_z)
        curlEy = (dEx_dz + self.Psi_Hy_z) - (dEz_dx + self.Psi_Hy_x)
        curlEz = (dEy_dx + self.Psi_Hz_x) - (dEx_dy + self.Psi_Hz_y)

        self.Hx -= (self.dt / (self.mu0 * self.MRxx)) * curlEx
        self.Hy -= (self.dt / (self.mu0 * self.MRyy)) * curlEy
        self.Hz -= (self.dt / (self.mu0 * self.MRzz)) * curlEz

    def _update_E(self):
        dHy_dz = (self.Hy - np.roll(self.Hy, 1, axis=2)) / self.dz
        dHz_dy = (self.Hz - np.roll(self.Hz, 1, axis=1)) / self.dy
        dHz_dx = (self.Hz - np.roll(self.Hz, 1, axis=0)) / self.dx
        dHx_dz = (self.Hx - np.roll(self.Hx, 1, axis=2)) / self.dz
        dHx_dy = (self.Hx - np.roll(self.Hx, 1, axis=1)) / self.dy
        dHy_dx = (self.Hy - np.roll(self.Hy, 1, axis=0)) / self.dx

        self.Psi_Ex_y = self.b_y * self.Psi_Ex_y + self.c_y * dHz_dy
        self.Psi_Ex_z = self.b_z * self.Psi_Ex_z + self.c_z * dHy_dz
        self.Psi_Ey_x = self.b_x * self.Psi_Ey_x + self.c_x * dHz_dx
        self.Psi_Ey_z = self.b_z * self.Psi_Ey_z + self.c_z * dHx_dz
        self.Psi_Ez_x = self.b_x * self.Psi_Ez_x + self.c_x * dHy_dx
        self.Psi_Ez_y = self.b_y * self.Psi_Ez_y + self.c_y * dHx_dy

        curlHx = (dHz_dy + self.Psi_Ex_y) - (dHy_dz + self.Psi_Ex_z)
        curlHy = (dHx_dz + self.Psi_Ey_z) - (dHz_dx + self.Psi_Ey_x)
        curlHz = (dHy_dx + self.Psi_Ez_x) - (dHx_dy + self.Psi_Ez_y)

        self.Ex += (self.dt / (self.eps0 * self.ERxx)) * curlHx
        self.Ey += (self.dt / (self.eps0 * self.ERyy)) * curlHy
        self.Ez += (self.dt / (self.eps0 * self.ERzz)) * curlHz

    # ---------- NF2FF ----------
    def nf2ff(self, near_field_box, theta, phi, r_obs=1.0):
        """
        Near-field to far-field (NF2FF) placeholder.
        near_field_box: dict with E and H samples on a closed surface.
        theta, phi: observation angles (radians) arrays or scalars.
        r_obs: observation radius.

        Returns (E_theta, E_phi) arrays. User can replace with full Stratton-Chu.
        """
        theta = np.asarray(theta, dtype=float)
        phi = np.asarray(phi, dtype=float)
        # TODO: compute true NF2FF integral; this placeholder returns zeros
        shape = np.broadcast(theta, phi).shape
        return np.zeros(shape, dtype=complex), np.zeros(shape, dtype=complex)

    # ---------- visualization ----------
    def plot_slice(self, component='Ez', axis='z', index=None, cmap='RdBu'):
        if axis not in ('x', 'y', 'z'):
            raise ValueError("axis must be 'x', 'y', or 'z'.")

        if index is None:
            index = {'x': self.Nx // 2, 'y': self.Ny // 2, 'z': self.Nz // 2}[axis]

        field = getattr(self, component)
        if axis == 'x':
            data = field[index, :, :]
            extent = [0, self.y_range, 0, self.z_range]
            xlabel, ylabel = 'y', 'z'
        elif axis == 'y':
            data = field[:, index, :]
            extent = [0, self.x_range, 0, self.z_range]
            xlabel, ylabel = 'x', 'z'
        else:
            data = field[:, :, index]
            extent = [0, self.x_range, 0, self.y_range]
            xlabel, ylabel = 'x', 'y'

        plt.figure()
        plt.imshow(data.T, origin='lower', extent=extent, cmap=cmap)
        plt.colorbar(label=component)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"{component} slice @ {axis}={index}")
        plt.tight_layout()
        plt.show()

    def plot_vector_3d(self, stride=4, scale=1.0):
        xs = np.arange(0, self.Nx, stride)
        ys = np.arange(0, self.Ny, stride)
        zs = np.arange(0, self.Nz, stride)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

        U = self.Ex[X, Y, Z]
        V = self.Ey[X, Y, Z]
        W = self.Ez[X, Y, Z]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(X, Y, Z, U, V, W, length=scale, normalize=True)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('E-field vector plot')
        plt.tight_layout()
        plt.show()
