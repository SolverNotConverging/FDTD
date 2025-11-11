import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm


class FDTD_2D:
    """
    2D TMz FDTD with:
      • multiple sources (point, line-soft, SF/TF)
      • PML (sigma profiles on edges)
      • save/load state (.npz) including multi-source + PML settings
      • animation that marks sources (red) and PML (black, alpha=0.3)
    """

    # ---------- construction ----------
    def __init__(self, x_range, y_range, Nx, Ny, f_max, Nt, f_min=None, dt=None):
        # constants
        self.eps0 = 8.85e-12
        self.mu0 = 4e-7 * np.pi
        self.c0 = 1 / np.sqrt(self.eps0 * self.mu0)
        self.eta0 = np.sqrt(self.mu0 / self.eps0)

        # grid
        self.x_range = float(x_range)
        self.y_range = float(y_range)
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.dx = self.x_range / self.Nx
        self.dy = self.y_range / self.Ny

        # time
        self.Nt = int(Nt)
        self.f_min = f_min if f_min is not None else None
        self.f_max = float(f_max)
        dt_cfl = np.sqrt(self.dx ** 2 + self.dy ** 2) / (2 * self.c0)
        dt_freq_sampling = 1.0 / (20 * self.f_max)
        self.dt = float(dt) if dt is not None else min(dt_cfl, dt_freq_sampling)

        # periodic directions list like ['x','y'] or []
        self.periodic = []

        # materials
        self.ERzz = np.ones((self.Nx, self.Ny))
        self.MRxx = np.ones((self.Nx, self.Ny))
        self.MRyy = np.ones((self.Nx, self.Ny))

        # fields
        self.Hx = np.zeros((self.Nx, self.Ny))
        self.Hy = np.zeros((self.Nx, self.Ny))
        self.Dz = np.zeros((self.Nx, self.Ny))
        self.Ez = np.zeros((self.Nx, self.Ny))

        # PML loss arrays
        self.sigma_hx = np.zeros((self.Nx, self.Ny))
        self.sigma_hy = np.zeros((self.Nx, self.Ny))
        self.sigma_dx = np.zeros((self.Nx, self.Ny))
        self.sigma_dy = np.zeros((self.Nx, self.Ny))

        # update coefficients
        self.mHx1 = np.zeros((self.Nx, self.Ny))
        self.mHx2 = np.zeros((self.Nx, self.Ny))
        self.mHx3 = np.zeros((self.Nx, self.Ny))
        self.mHy1 = np.zeros((self.Nx, self.Ny))
        self.mHy2 = np.zeros((self.Nx, self.Ny))
        self.mHy3 = np.zeros((self.Nx, self.Ny))
        self.mDz1 = np.zeros((self.Nx, self.Ny))
        self.mDz2 = np.zeros((self.Nx, self.Ny))
        self.mDz3 = np.zeros((self.Nx, self.Ny))
        self.mEz1 = np.zeros((self.Nx, self.Ny))

        # curls + integrals
        self.CEx = np.zeros((self.Nx, self.Ny))
        self.CEy = np.zeros((self.Nx, self.Ny))
        self.CHz = np.zeros((self.Nx, self.Ny))
        self.ICEx = np.zeros((self.Nx, self.Ny))
        self.ICEy = np.zeros((self.Nx, self.Ny))
        self.IDz = np.zeros((self.Nx, self.Ny))

        # multi-source list ---
        # each source is a dict with keys:
        #   kind: 'point' | 'line-soft' | 'sftf-x' | 'sftf-y'
        #   ix0, ix1, iy0, iy1 (ints; for points, ix0,iy0 used; for lines, spans are used)
        #   amplitude, t0, tw, f_min (or None), f_max
        #   direction: '+x'/'-x'/' +y'/'-y' (used by SF/TF if needed later)
        self.sources = []
        self.avg_freqs = []  # one per source (spectral centroid for info/diagnostics)

        self.monitor = []
        self.monitor_results = []

        # --- NEW: store PML parameters as attributes (for save/load & plotting) ---
        self.pml_width = None
        self.sigma_max = None
        self.pml_order = 3
        self.pml_direction = 'xy'  # 'x'|'y'|'xy'

        # ---------- coefficients ----------

    def _init_m(self):
        mHx0 = 1 / self.dt + self.sigma_hy / (2 * self.eps0)
        self.mHx1 = (1 / self.dt - self.sigma_hy / (2 * self.eps0)) / mHx0
        self.mHx2 = -self.c0 / self.MRxx / mHx0
        self.mHx3 = -(self.c0 * self.dt / self.eps0) * self.sigma_hx / self.MRxx / mHx0

        mHy0 = 1 / self.dt + self.sigma_hx / (2 * self.eps0)
        self.mHy1 = (1 / self.dt - self.sigma_hx / (2 * self.eps0)) / mHy0
        self.mHy2 = -self.c0 / self.MRyy / mHy0
        self.mHy3 = -(self.c0 * self.dt / self.eps0) * self.sigma_hy / self.MRyy / mHy0

        mDz0 = 1 / self.dt + (self.sigma_dx + self.sigma_dy) / (2 * self.eps0) + self.sigma_dx * self.sigma_dy * (
                self.dt / 4 / self.eps0 ** 2)
        mDz1 = (1 / self.dt) - (self.sigma_dx + self.sigma_dy) / (2 * self.eps0) - self.sigma_dx * self.sigma_dy * (
                self.dt / 4 / self.eps0 ** 2)
        self.mDz1 = mDz1 / mDz0
        self.mDz2 = self.c0 / mDz0
        self.mDz3 = -(self.dt / self.eps0 ** 2) * self.sigma_dx * self.sigma_dy / mDz0

        self.mEz1 = 1 / self.ERzz

    # ---------- spatial curls ----------
    def calculate_CE(self):
        # identical to your implementations (periodic variants)
        per_x = hasattr(self, "periodic") and ('x' in self.periodic)
        per_y = hasattr(self, "periodic") and ('y' in self.periodic)

        if per_y:
            for nx in range(self.Nx):
                for ny in range(self.Ny - 1):
                    self.CEx[nx, ny] = (self.Ez[nx, ny + 1] - self.Ez[nx, ny]) / self.dy
                self.CEx[nx, self.Ny - 1] = (self.Ez[nx, 0] - self.Ez[nx, self.Ny - 1]) / self.dy
        else:
            for nx in range(self.Nx):
                for ny in range(self.Ny - 1):
                    self.CEx[nx, ny] = (self.Ez[nx, ny + 1] - self.Ez[nx, ny]) / self.dy
                self.CEx[nx, self.Ny - 1] = (0 - self.Ez[nx, self.Ny - 1]) / self.dy

        if per_x:
            for ny in range(self.Ny):
                for nx in range(self.Nx - 1):
                    self.CEy[nx, ny] = -(self.Ez[nx + 1, ny] - self.Ez[nx, ny]) / self.dx
                self.CEy[self.Nx - 1, ny] = -(self.Ez[0, ny] - self.Ez[self.Nx - 1, ny]) / self.dx
        else:
            for ny in range(self.Ny):
                for nx in range(self.Nx - 1):
                    self.CEy[nx, ny] = -(self.Ez[nx + 1, ny] - self.Ez[nx, ny]) / self.dx
                self.CEy[self.Nx - 1, ny] = -(0 - self.Ez[self.Nx - 1, ny]) / self.dx

    def calculate_ICE(self):
        self.ICEx += self.CEx
        self.ICEy += self.CEy

    def update_H(self):
        self.Hx = self.mHx1 * self.Hx + self.mHx2 * self.CEx + self.mHx3 * self.ICEx
        self.Hy = self.mHy1 * self.Hy + self.mHy2 * self.CEy + self.mHy3 * self.ICEy

    def calculate_CH(self):
        per_x = hasattr(self, "periodic") and ('x' in self.periodic)
        per_y = hasattr(self, "periodic") and ('y' in self.periodic)

        if (not per_x) and (not per_y):
            self.CHz[0, 0] = (self.Hy[0, 0] - 0.0) / self.dx - (self.Hx[0, 0] - 0.0) / self.dy
            for nx in range(1, self.Nx):
                self.CHz[nx, 0] = (self.Hy[nx, 0] - self.Hy[nx - 1, 0]) / self.dx - (self.Hx[nx, 0] - 0.0) / self.dy
            for ny in range(1, self.Ny):
                self.CHz[0, ny] = (self.Hy[0, ny] - 0.0) / self.dx - (self.Hx[0, ny] - self.Hx[0, ny - 1]) / self.dy
                for nx in range(1, self.Nx):
                    self.CHz[nx, ny] = (self.Hy[nx, ny] - self.Hy[nx - 1, ny]) / self.dx - (
                            self.Hx[nx, ny] - self.Hx[nx, ny - 1]) / self.dy

        elif per_x and (not per_y):
            self.CHz[0, 0] = (self.Hy[0, 0] - self.Hy[self.Nx - 1, 0]) / self.dx - (self.Hx[0, 0] - 0.0) / self.dy
            for nx in range(1, self.Nx):
                self.CHz[nx, 0] = (self.Hy[nx, 0] - self.Hy[nx - 1, 0]) / self.dx - (self.Hx[nx, 0] - 0.0) / self.dy
            for ny in range(1, self.Ny):
                self.CHz[0, ny] = (self.Hy[0, ny] - self.Hy[self.Nx - 1, ny]) / self.dx - (
                        self.Hx[0, ny] - self.Hx[0, ny - 1]) / self.dy
                for nx in range(1, self.Nx):
                    self.CHz[nx, ny] = (self.Hy[nx, ny] - self.Hy[nx - 1, ny]) / self.dx - (
                            self.Hx[nx, ny] - self.Hx[nx, ny - 1]) / self.dy

        elif (not per_x) and per_y:
            self.CHz[0, 0] = (self.Hy[0, 0] - 0.0) / self.dx - (self.Hx[0, 0] - self.Hx[0, self.Ny - 1]) / self.dy
            for nx in range(1, self.Nx):
                self.CHz[nx, 0] = (self.Hy[nx, 0] - self.Hy[nx - 1, 0]) / self.dx - (
                        self.Hx[nx, 0] - self.Hx[nx, self.Ny - 1]) / self.dy
            for ny in range(1, self.Ny):
                self.CHz[0, ny] = (self.Hy[0, ny] - 0.0) / self.dx - (self.Hx[0, ny] - self.Hx[0, ny - 1]) / self.dy
                for nx in range(1, self.Nx):
                    self.CHz[nx, ny] = (self.Hy[nx, ny] - self.Hy[nx - 1, ny]) / self.dx - (
                            self.Hx[nx, ny] - self.Hx[nx, ny - 1]) / self.dy

        else:
            self.CHz[0, 0] = (self.Hy[0, 0] - self.Hy[self.Nx - 1, 0]) / self.dx - (
                    self.Hx[0, 0] - self.Hx[0, self.Ny - 1]) / self.dy
            for nx in range(1, self.Nx):
                self.CHz[nx, 0] = (self.Hy[nx, 0] - self.Hy[nx - 1, 0]) / self.dx - (
                        self.Hx[nx, 0] - self.Hx[nx, self.Ny - 1]) / self.dy
            for ny in range(1, self.Ny):
                self.CHz[0, ny] = (self.Hy[0, ny] - self.Hy[self.Nx - 1, ny]) / self.dx - (
                        self.Hx[0, ny] - self.Hx[0, ny - 1]) / self.dy
                for nx in range(1, self.Nx):
                    self.CHz[nx, ny] = (self.Hy[nx, ny] - self.Hy[nx - 1, ny]) / self.dx - (
                            self.Hx[nx, ny] - self.Hx[nx, ny - 1]) / self.dy

    def calculate_ID(self):
        self.IDz += self.Dz

    def update_D(self):
        self.Dz = self.mDz1 * self.Dz + self.mDz2 * self.CHz + self.mDz3 * self.IDz

    def update_E(self):
        self.Ez = self.mEz1 * self.Dz

    # ---------- geometry helpers ----------
    def add_rectangle(self, ER, MR, x_position, y_position):
        if isinstance(ER, (list, tuple, np.ndarray)) and len(ER) == 3:
            ERzz_obj = float(ER[2])
        else:
            ERzz_obj = float(ER)
        if isinstance(MR, (list, tuple, np.ndarray)) and len(MR) == 3:
            MRxx_obj = float(MR[0])
            MRyy_obj = float(MR[1])
        else:
            MRxx_obj = float(MR)
            MRyy_obj = float(MR)

        def edge_to_m(val, axis='x'):
            if isinstance(val, (int, np.integer)): return (val * (self.dx if axis == 'x' else self.dy))
            return float(val)

        def interval_overlap(a0, a1, b0, b1):
            lo = max(a0, b0)
            hi = min(a1, b1)
            return max(0.0, hi - lo)

        if not (isinstance(x_position, (list, tuple)) and len(x_position) == 2): raise TypeError(
            "x_position must be (x0,x1).")
        if not (isinstance(y_position, (list, tuple)) and len(y_position) == 2): raise TypeError(
            "y_position must be (y0,y1).")

        x0m = edge_to_m(x_position[0], 'x')
        x1m = edge_to_m(x_position[1], 'x')
        y0m = edge_to_m(y_position[0], 'y')
        y1m = edge_to_m(y_position[1], 'y')
        if x1m < x0m: x0m, x1m = x1m, x0m
        if y1m < y0m: y0m, y1m = y1m, y0m
        x0m = max(0.0, min(self.x_range, x0m))
        x1m = max(0.0, min(self.x_range, x1m))
        y0m = max(0.0, min(self.y_range, y0m))
        y1m = max(0.0, min(self.y_range, y1m))
        if (x1m <= x0m) or (y1m <= y0m): return

        dx, dy = self.dx, self.dy
        Nx, Ny = self.Nx, self.Ny
        i_min = max(0, int(np.floor(x0m / dx)))
        i_max = min(Nx - 1, int(np.ceil(x1m / dx)) - 1)
        j_min = max(0, int(np.floor(y0m / dy)))
        j_max = min(Ny - 1, int(np.ceil(y1m / dy)) - 1)

        for i in range(i_min, i_max + 1):
            cell_x0 = i * dx
            cell_x1 = (i + 1) * dx
            fx = interval_overlap(x0m, x1m, cell_x0, cell_x1) / dx
            if fx == 0.0: continue
            for j in range(j_min, j_max + 1):
                cell_y0 = j * dy
                cell_y1 = (j + 1) * dy
                fy = interval_overlap(y0m, y1m, cell_y0, cell_y1) / dy
                if fy == 0.0: continue
                f = fx * fy
                self.ERzz[i, j] = (1.0 - f) * self.ERzz[i, j] + f * ERzz_obj
                self.MRxx[i, j] = (1.0 - f) * self.MRxx[i, j] + f * MRxx_obj
                self.MRyy[i, j] = (1.0 - f) * self.MRyy[i, j] + f * MRyy_obj

    def add_circle(self, ER, MR, center, radius, nsub=6):
        """
        Add a (possibly anisotropic) circular object with subpixel edge smoothing.

        ER: float (isotropic) or (ERxx, ERyy, ERzz). TMz uses ERzz.
        MR: float (isotropic) or (MRxx, MRyy, MRzz). TMz uses MRxx, MRyy.
        center: (cx, cy) where each element is int (edge index) or float (meters).
        radius: float, meters.
        nsub: supersamples per axis (nsub x nsub per cell) for area fraction.
        """
        # --- parse materials (TMz: Ez/Dz along z, Hx/Hy in-plane) ---
        if isinstance(ER, (list, tuple, np.ndarray)) and len(ER) == 3:
            ERzz_obj = float(ER[2])
        else:
            ERzz_obj = float(ER)

        if isinstance(MR, (list, tuple, np.ndarray)) and len(MR) == 3:
            MRxx_obj = float(MR[0])
            MRyy_obj = float(MR[1])
        else:
            MRxx_obj = float(MR)
            MRyy_obj = float(MR)

        # --- parse geometry ---
        if not (isinstance(center, (list, tuple)) and len(center) == 2):
            raise TypeError("center must be a 2-tuple/list (int edge index or float meters).")
        if not isinstance(radius, (float, int)):
            raise TypeError("radius must be a float/int in meters.")
        radius = float(radius)
        if radius <= 0.0:
            return  # nothing to do

        def to_m(val, axis='x'):
            if isinstance(val, (int, np.integer)):
                return float(val) * (self.dx if axis == 'x' else self.dy)  # edge index → meters
            else:
                return float(val)

        cx = to_m(center[0], 'x')
        cy = to_m(center[1], 'y')

        # Clip center if someone passes slightly outside
        cx = max(0.0, min(self.x_range, cx))
        cy = max(0.0, min(self.y_range, cy))

        dx, dy = self.dx, self.dy
        Nx, Ny = self.Nx, self.Ny

        # --- bounding box in indices (conservative) ---
        x0 = max(0.0, cx - radius)
        x1 = min(self.x_range, cx + radius)
        y0 = max(0.0, cy - radius)
        y1 = min(self.y_range, cy + radius)

        i_min = max(0, int(np.floor(x0 / dx)))
        i_max = min(Nx - 1, int(np.ceil(x1 / dx)) - 1)
        j_min = max(0, int(np.floor(y0 / dy)))
        j_max = min(Ny - 1, int(np.ceil(y1 / dy)) - 1)

        if (i_max < i_min) or (j_max < j_min):
            return  # fully outside

        # --- supersampling grid offsets inside a cell ---
        # place nsub sample points uniformly within each cell
        # offsets measured from cell corner (x0,y0) to sample centers
        if nsub < 1:
            nsub = 1
        sx = (np.arange(nsub) + 0.5) * (dx / nsub)
        sy = (np.arange(nsub) + 0.5) * (dy / nsub)

        r2 = radius * radius
        inv_n2 = 1.0 / (nsub * nsub)

        # --- paint with area-fraction mixing ---
        for i in range(i_min, i_max + 1):
            cell_x0 = i * dx
            for j in range(j_min, j_max + 1):
                cell_y0 = j * dy

                # count how many subsamples inside the circle
                inside = 0
                # vectorize over x for a small speedup
                xs = cell_x0 + sx  # shape (nsub,)
                ys = cell_y0 + sy  # shape (nsub,)
                for yy in ys:
                    # (xs - cx)^2 + (yy - cy)^2 <= r^2
                    inside += np.count_nonzero((xs - cx) * (xs - cx) + (yy - cy) * (yy - cy) <= r2)

                f = inside * inv_n2  # area fraction in this cell
                if f <= 0.0:
                    continue

                self.ERzz[i, j] = (1.0 - f) * self.ERzz[i, j] + f * ERzz_obj
                self.MRxx[i, j] = (1.0 - f) * self.MRxx[i, j] + f * MRxx_obj
                self.MRyy[i, j] = (1.0 - f) * self.MRyy[i, j] + f * MRyy_obj

    # ---------- PML ----------
    def add_PML(self, pml_width, sigma_max, order=3, direction='xy'):
        # remember user params
        self.pml_width = pml_width
        self.sigma_max = float(sigma_max)
        self.pml_order = int(order)
        self.pml_direction = str(direction)

        # convert width to cells
        if isinstance(pml_width, int):
            npx = npy = max(1, int(pml_width))
        elif isinstance(pml_width, float):
            npx = max(1, int(np.ceil(pml_width / self.dx)))
            npy = max(1, int(np.ceil(pml_width / self.dy)))
        else:
            raise TypeError("pml_width must be int (cells) or float (meters).")

        if direction == 'x':
            npy = 0
        elif direction == 'y':
            npx = 0
        elif direction != 'xy':
            raise TypeError("direction must be 'x', 'y', or 'xy'.")

        # build profiles (same logic as your originals)
        for i in range(npx):
            prof_x = self.eps0 / (2 * self.dt) * sigma_max * ((npx - i) / npx) ** order
            self.sigma_hx[i, :] = prof_x
            self.sigma_hx[-i - 1, :] = prof_x
            self.sigma_dx[i, :] = prof_x
            self.sigma_dx[-i - 1, :] = prof_x

        for i in range(npy):
            prof_y = self.eps0 / (2 * self.dt) * sigma_max * ((npy - i) / npy) ** order
            self.sigma_hy[:, i] = prof_y
            self.sigma_hy[:, -i - 1] = prof_y
            self.sigma_dy[:, i] = prof_y
            self.sigma_dy[:, -i - 1] = prof_y

    # ---------- source helpers ----------
    def _g(self, s, t):
        """
        Time waveform for a single source 's' at time t (scalar or ndarray).

        Cases:
          • f_min is None  -> Gaussian pulse (centered at t0, width tw)
          • f_min == f_max -> ramped continuous sinusoid at f0 (no Gaussian)
          • otherwise      -> sinusoid * Gaussian (band-limited pulse)
        """
        amp = s["amplitude"]
        t0 = s["t0"]
        tw = s["tw"]
        fmin = s["f_min"]
        fmax = s["f_max"]

        # ensure numpy ops for both scalar and array t
        t = np.asarray(t, dtype=float)

        if fmin is None:
            # Gaussian pulse
            return amp * np.exp(-((t - t0) / tw) ** 2)

        if np.isclose(fmin, fmax):
            # Ramped continuous sinusoid (no Gaussian envelope)
            f0 = float(fmax)
            # Smooth cubic-on ramp; ~5 cycles to near-full amplitude (floor to >= 5*dt)
            Tr = max(1 / max(f0, 1e-30), 1 * self.dt)
            # Start ramping at t0 so you can time-align the tone with other sources
            tau = np.maximum(t - t0, 0.0)
            ramp = 1.0 - np.exp(-(tau / Tr) ** 3)
            return amp * ramp * np.sin(2 * np.pi * f0 * (t - t0))

        # Band-limited pulse: sinusoid under a Gaussian envelope
        f0 = 0.5 * (fmin + fmax)
        return amp * np.sin(2 * np.pi * f0 * (t - t0)) * np.exp(-((t - t0) / tw) ** 2)

    def _spec_centroid(self, waveform):
        freq = np.fft.fftfreq(len(waveform), d=self.dt)
        S = np.fft.fft(waveform)
        pos = freq >= 0
        f = freq[pos]
        mag = np.abs(S[pos]) + 1e-30
        return float(np.sum(f * mag) / np.sum(mag))

    def _wg_modes_y(self, ix0, ix1, iy, f_center, num_modes=4, guess=None, amplitude=1.0):
        import numpy as np
        from scipy.sparse import diags as spdiags
        from scipy.sparse.linalg import eigs

        lo, hi = (min(ix0, ix1), max(ix0, ix1))
        Nx = int(max(1, hi - lo))
        if Nx < 2:
            raise ValueError("waveguide-y: x-span too small; need at least 2 cells.")

        # constants
        k0 = 2.0 * np.pi * float(f_center) / self.c0
        dx = self.dx

        # material vectors along the slice
        ER_vec = np.asarray(self.ERzz[lo:hi, iy], dtype=float)
        MRxx_vec = np.asarray(self.MRxx[lo:hi, iy], dtype=float)
        MRyy_vec = np.asarray(self.MRyy[lo:hi, iy], dtype=float)

        # sparse diagonals (NOTE: shape must be a tuple)
        ERzz_diag = spdiags(ER_vec, 0, shape=(Nx, Nx))
        MRxx_inv = spdiags(1.0 / MRxx_vec, 0, shape=(Nx, Nx))
        MRyy_inv = spdiags(1.0 / MRyy_vec, 0, shape=(Nx, Nx))

        # your difference operators (exact layout and scaling)
        d_plus = np.ones(Nx)
        d_minus = -np.ones(Nx)
        # DEX: offsets [1, 0] / (dx*k0)  -> upper diag = +1, main = -1
        DEX = spdiags([d_plus, d_minus], [1, 0], shape=(Nx, Nx)) / (dx * k0)
        # DHX: offsets [0, 1] / (dx*k0)  -> main = +1, lower = -1
        DHX = spdiags([d_plus, d_minus], [0, -1], shape=(Nx, Nx)) / (dx * k0)

        # operators
        A = ERzz_diag + DHX @ (MRxx_inv @ DEX)
        B = MRyy_inv

        # shift guess near n_core^2
        if guess is None:
            n_slice = np.sqrt(ER_vec * 0.5 * (MRxx_vec + MRyy_vec))
            n_guess = float(np.max(n_slice))
            guess = max(n_guess ** 2, 1.0)

        k = int(max(1, num_modes))
        evals, evecs = eigs(A, M=B, k=k, sigma=guess)  # complex in general

        # sort by descending Re(n_eff)
        n_eff = np.sqrt(np.maximum(evals.real, 0.0))
        order = np.argsort(-n_eff)
        evecs = evecs[:, order]
        n_eff = n_eff[order]

        # ---------- normalize & scale (tutorial-consistent) ----------
        Ez_modes = []
        Hx_modes = []

        for m in range(evecs.shape[1]):
            Ez = evecs[:, m]

            kmax = np.argmax(np.abs(Ez))
            phase = np.angle(Ez[kmax])
            Ez = (Ez * np.exp(-1j * phase)).real

            # Normalize Ez to max|.| = 1 (shape only)
            Ez = Ez / (np.max(np.abs(Ez)) + 1e-30)

            # Tutorial relation: h_x = - n_eff * mu_xx^{-1} * e_z
            Hx = -(n_eff[m]) * (MRxx_inv @ Ez)
            # MRxx_inv is a sparse diagonal; ensure dense vector
            Hx = Hx.A.squeeze() if hasattr(Hx, "A") else np.asarray(Hx).squeeze()
            Hx = Hx.real

            Ez_modes.append(Ez)
            Hx_modes.append(Hx)

        Ez_modes = np.asarray(Ez_modes)
        Hx_modes = np.asarray(Hx_modes)

        return np.asarray(Ez_modes), np.asarray(Hx_modes), np.asarray(n_eff, dtype=float)

    # ---------- public API: add_source ----------
    def add_source(self, kind, x, y, amplitude=1.0, t0=None, tw=None, f_min=None, f_max=None, mode_index=1,
                   modes_to_show=4, eig_guess=None, is_show=True, ):

        """
        Add a source.

        kind:
          'point'     : soft point into Dz at (x,y)
          'line-soft' : soft line into Dz; give x=(ix0,ix1) & y=j or y=(j0,j1) & x=i
          'sftf-x'    : TF/SF boundary with **normal along x**  → vertical line at fixed x, spanning y
          'sftf-y'    : TF/SF boundary with **normal along y**  → horizontal line at fixed y, spanning x
        x, y:
          Either ints (indices) or floats (meters). For spans, pass (start, end).
           num_modes, mode_index and guess are used only for 'waveguide' mode
        """

        k = kind.lower()
        if k not in ('point', 'line-soft', 'sftf-x', 'sftf-y', 'waveguide-y'):
            raise ValueError("kind must be 'point', 'line-soft', 'sftf-x', 'sftf-y', 'waveguide-y'.")

        # normalize frequency parameters
        fmin = f_min if f_min is not None else self.f_min
        fmax = f_max if f_max is not None else self.f_max

        # window params
        if fmin is None:
            tw_default = 0.5 / fmax
            t0_default = 4 * tw_default

        elif fmin == fmax:
            tw_default = 0.25 / fmax
            t0_default = 1 * tw_default
        else:
            tw_default = 2.0 / fmax
            t0_default = 4 * tw_default

        tw = tw if tw is not None else tw_default
        t0 = t0 if t0 is not None else t0_default

        def to_index_x(val):
            if isinstance(val, (int, np.integer)): return int(np.clip(val, 0, self.Nx))
            return int(np.clip(np.round(float(val) / self.dx), 0, self.Nx))

        def to_index_y(val):
            if isinstance(val, (int, np.integer)): return int(np.clip(val, 0, self.Ny))
            return int(np.clip(np.round(float(val) / self.dy), 0, self.Ny))

        def parse_span(arg, to_index):
            if isinstance(arg, (list, tuple, np.ndarray)):
                a0 = to_index(arg[0])
                a1 = to_index(arg[1])
                if a1 < a0: a0, a1 = a1, a0
                return int(a0), int(a1)
            else:
                a = to_index(arg)
                return int(a), int(a)

        ix0, ix1 = parse_span(x, to_index_x)
        iy0, iy1 = parse_span(y, to_index_y)

        s = dict(
            kind=k,
            ix0=int(ix0), ix1=int(ix1),
            iy0=int(iy0), iy1=int(iy1),
            amplitude=float(amplitude), t0=float(t0), tw=float(tw),
            f_min=(None if fmin is None else float(fmin)),
            f_max=float(fmax),
        )

        # optional preview
        if is_show:
            import matplotlib.pyplot as plt
            t = np.arange(0, self.Nt * self.dt, self.dt)
            g = self._g(s, t)
            fig, axs = plt.subplots(2, 1, figsize=(6, 6))
            axs[0].plot(t * 1e9, g)
            axs[0].set_xlabel('Time (ns)')
            axs[0].set_ylabel('Amplitude')
            axs[0].set_title('Source g(t)')
            freq = np.fft.fftfreq(len(t), d=self.dt)
            S = np.abs(np.fft.fft(g))
            pos = freq >= 0
            axs[1].plot(freq[pos] / 1e9, S[pos])
            axs[1].set_xlim(0, 2 * fmax / 1e9)
            axs[1].set_xlabel('Frequency (GHz)')
            axs[1].set_ylabel('Magnitude')
            axs[1].set_title('Spectrum')
            plt.tight_layout()
            plt.show()

        tt = np.arange(0, self.Nt * self.dt, self.dt)
        self.avg_freqs.append(self._spec_centroid(self._g(s, tt)))

        # --- waveguide port (horizontal, normal = y) ---
        if k == 'waveguide-y':
            lo, hi = (min(ix0, ix1), max(ix0, ix1))
            iy_line = iy0
            # choose a reasonable center frequency
            if f_min is not None and f_max is not None:
                f_center = 0.5 * (f_min + f_max)
            elif f_max is not None:
                f_center = f_max
            else:
                f_center = self.f_max

            Ez_modes, Hx_modes, n_effs = self._wg_modes_y(
                lo, hi, iy_line, f_center,
                num_modes=max(1, int(modes_to_show)),
                guess=eig_guess, amplitude=float(amplitude)
            )
            # Visualize only the Ez mode profiles; title shows n_eff
            # Visualize Ez and Hx profiles; title shows n_eff
            if is_show:
                import matplotlib.pyplot as plt
                lo, hi = (min(ix0, ix1), max(ix0, ix1))
                x_axis = (np.arange(lo, hi) + 0.5) * self.dx  # cell centers

                rows = min(Ez_modes.shape[0], int(modes_to_show))
                fig, axs = plt.subplots(rows, 1, figsize=(8, 2.6 * rows), sharex=True)
                if rows == 1: axs = [axs]

                for m in range(rows):
                    ax1 = axs[m]
                    ax2 = ax1.twinx()
                    ax1.plot(x_axis, Ez_modes[m], linewidth=1.6, label='Ez')
                    ax2.plot(x_axis, Hx_modes[m], linestyle='--', linewidth=1.2, label='Hx')

                    ax1.set_ylabel('Ez (arb.)')
                    ax2.set_ylabel('Hx (arb.)')
                    ax1.set_title(f'mode {m + 1}: n_eff = {n_effs[m]:.6f}')
                    ax1.grid(True, alpha=0.25)

                    # compact combined legend
                    lines, labels = [], []
                    for a in (ax1, ax2):
                        l, lab = a.get_legend_handles_labels()
                        lines += l
                        labels += lab
                    ax1.legend(lines, labels, loc='upper right')

                axs[-1].set_xlabel('x (m)')
                fig.suptitle(f'waveguide-y port at y={iy_line}')
                fig.tight_layout()
                plt.show()

            # select which mode to inject (1-based)
            mi = max(1, int(mode_index)) - 1
            mi = min(mi, Ez_modes.shape[0] - 1)
            s['Ez_src'] = Ez_modes[mi]
            s['Hx_src'] = Hx_modes[mi]
            s['n_eff'] = float(n_effs[mi])

        self.sources.append(s)

        # --- Line Monitor ---

    def add_line_monitor(self, x, y, t=None):
        """
        Add a 1D monitor aligned to the grid:
          - horizontal: x = (x0, x1), y = y0
          - vertical  : x = x0,       y = (y0, y1)
        x, y, t can be indices or meters/seconds. Spans can be 2-tuples/lists.
        Time indices are [it0, it1) (end-exclusive).
        """

        def _to_index(val, step, N):
            # Accept index or physical value
            if isinstance(val, (int, np.integer)):
                return int(np.clip(val, 0, N))
            return int(np.clip(np.round(float(val) / step), 0, N))

        def _span(arg, step, N):
            if isinstance(arg, (list, tuple, np.ndarray)):
                a0 = _to_index(arg[0], step, N)
                a1 = _to_index(arg[1], step, N)
                if a1 < a0: a0, a1 = a1, a0
                return int(a0), int(a1)
            a = _to_index(arg, step, N)
            return int(a), int(a)

        # parse spatial spans
        ix0, ix1 = _span(x, self.dx, self.Nx)
        iy0, iy1 = _span(y, self.dy, self.Ny)

        # exactly one span must be a line (nonzero length)
        is_h = (ix0 != ix1) and (iy0 == iy1)  # horizontal
        is_v = (ix0 == ix1) and (iy0 != iy1)  # vertical
        if not (is_h or is_v):
            raise ValueError("Provide a horizontal line (x=(x0,x1), y=y0) or a vertical line (x=x0, y=(y0,y1)).")

        # time span (end-exclusive); default: whole run
        if t is None:
            it0, it1 = 0, self.Nt
        else:
            it0, it1 = _span(t, self.dt, self.Nt)

        self.monitor.append({
            "ix0": ix0, "ix1": ix1,
            "iy0": iy0, "iy1": iy1,
            "it0": it0, "it1": it1,
            "orientation": "horizontal" if is_h else "vertical",
        })

    # ---------- main loop ----------

    def run(self, record_stride=1, is_include_history=True):
        self._init_m()
        self.is_include_history = is_include_history
        Nx, Ny = self.Nx, self.Ny

        # recording metadata
        self.record_stride = int(record_stride)

        if self.is_include_history:
            Nt_rec = (self.Nt + self.record_stride - 1) // self.record_stride
            self.Nt_rec = int(Nt_rec)
            # allocate histories with compact dtype (float32) to reduce RAM
            dtype_hist = self.Hx.dtype  # or: np.float32
            self.Hx_history = np.zeros((Nt_rec, Nx, Ny), dtype=dtype_hist)
            self.Hy_history = np.zeros((Nt_rec, Nx, Ny), dtype=dtype_hist)
            self.Ez_history = np.zeros((Nt_rec, Nx, Ny), dtype=dtype_hist)
            self.Dz_history = np.zeros((Nt_rec, Nx, Ny), dtype=dtype_hist)
            rec_idx = 0
        else:
            # no history arrays in monitor-only mode
            self.Nt_rec = 0
            rec_idx = None

        # --- monitors: prepare buffers (use existing 'orientation' and unpack with **) ---
        monitor_results = []

        for m in self.monitor:
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
                    **m,  # inherit all monitor keys
                    "Ez": np.empty((Tm, L), dtype=self.Ez.dtype),
                    "Hx": np.empty((Tm, L), dtype=self.Hx.dtype),
                    "Hy": np.empty((Tm, L), dtype=self.Hy.dtype),
                    "_slx": slice(ix0, ix1),  # precomputed x-span
                    "_y": iy0,  # fixed y
                }
            else:  # vertical
                L = iy1 - iy0
                Tm = max(0, it1 - it0)
                if L <= 0 or Tm <= 0:
                    continue
                buf = {
                    **m,
                    "Ez": np.empty((Tm, L), dtype=self.Ez.dtype),
                    "Hx": np.empty((Tm, L), dtype=self.Hx.dtype),
                    "Hy": np.empty((Tm, L), dtype=self.Hy.dtype),
                    "_x": ix0,  # fixed x
                    "_sly": slice(iy0, iy1),  # precomputed y-span
                }

            monitor_results.append(buf)

        for t_index in tqdm(range(self.Nt), desc="FDTD simulation", unit="step"):
            # E-curl
            self.calculate_CE()

            # --- SF/TF E injection (by normal) ---
            for s in self.sources:
                if s["kind"] == 'sftf-y':
                    # horizontal line (normal = y), at y = iy0, span in x
                    y = s["iy0"]
                    i0, i1 = s["ix0"], s["ix1"]
                    E_src = self._g(s, t_index * self.dt)
                    for i in range(min(i0, i1), max(i0, i1)):
                        if 0 <= y - 1 < self.Ny:
                            self.CEx[i, y - 1] += (1.0 / self.dy) * E_src
                elif s["kind"] == 'sftf-x':
                    # vertical line (normal = x), at x = ix0, span in y
                    x = s["ix0"]
                    j0, j1 = s["iy0"], s["iy1"]
                    E_src = self._g(s, t_index * self.dt)
                    for j in range(min(j0, j1), max(j0, j1)):
                        if 0 <= x - 1 < self.Nx:
                            self.CEy[x - 1, j] -= (1.0 / self.dx) * E_src

                # E injection (waveguide-y)
                elif s['kind'] == 'waveguide-y':
                    y = s["iy0"]
                    i0, i1 = s["ix0"], s["ix1"]
                    lo, hi = (min(i0, i1), max(i0, i1))
                    E_src = self._g(s, t_index * self.dt)
                    for i in range(lo, hi):
                        if 0 <= y - 1 < self.Ny:
                            self.CEx[i, y - 1] += (1.0 / self.dy) * E_src * s["Ez_src"][i - lo]

            # integrate CE and update H
            self.calculate_ICE()
            self.update_H()

            # H-curl
            self.calculate_CH()

            # --- SF/TF H injection (by normal) ---
            for s in self.sources:
                if s["kind"] == 'sftf-y':
                    # horizontal TF/SF (normal = y) → use dy/2, dt/2 stagger
                    H_src = -self._g(s, t_index * self.dt + self.dy / (2 * self.c0) + self.dt / 2.0)
                    y = s["iy0"]
                    i0, i1 = s["ix0"], s["ix1"]
                    for i in range(min(i0, i1), max(i0, i1)):
                        if 0 <= y < self.Ny:
                            self.CHz[i, y] -= (1.0 / self.dy) * H_src

                elif s["kind"] == 'sftf-x':
                    # vertical TF/SF (normal = x) → use dx/2, dt/2 stagger
                    H_src = -self._g(s, t_index * self.dt + self.dx / (2 * self.c0) + self.dt / 2.0)
                    x = s["ix0"]
                    j0, j1 = s["iy0"], s["iy1"]
                    for j in range(min(j0, j1), max(j0, j1)):
                        if 0 <= x < self.Nx:
                            self.CHz[x, j] += (1.0 / self.dx) * H_src

                # H injection (waveguide-y)
                elif s["kind"] == 'waveguide-y':
                    n_eff = s["n_eff"]
                    H_src = -self._g(s, t_index * self.dt + self.dy * n_eff / (2 * self.c0) + self.dt / 2.0)
                    y = s["iy0"]
                    i0, i1 = s["ix0"], s["ix1"]
                    lo, hi = (min(i0, i1), max(i0, i1))
                    for i in range(lo, hi):
                        if 0 <= y < self.Ny:
                            self.CHz[i, y] -= (1.0 / self.dy) * H_src * s["Hx_src"][i - lo]

            # integrate CH and update D
            self.calculate_ID()
            self.update_D()

            # --- soft sources (point/line-soft) into Dz ---
            t_now = t_index * self.dt
            for s in self.sources:
                if s["kind"] == 'point':
                    i, j = s["ix0"], s["iy0"]
                    self.Dz[i, j] += self._g(s, t_now)  # :contentReference[oaicite:9]{index=9}
                elif s["kind"] == 'line-soft':
                    # If ix span -> horizontal line at y=iy0; else if iy span -> vertical line at x=ix0
                    if s["ix0"] != s["ix1"]:
                        y = s["iy0"]
                        for i in range(min(s["ix0"], s["ix1"]), max(s["ix0"], s["ix1"])):
                            self.Dz[i, y] += self._g(s, t_now)
                    else:
                        x = s["ix0"]
                        for j in range(min(s["iy0"], s["iy1"]), max(s["iy0"], s["iy1"])):
                            self.Dz[x, j] += self._g(s, t_now)

            # update E
            self.update_E()

            # --- capture monitors at this step (no squeeze; direct 1D writes) ---
            for buf in monitor_results:
                if buf["it0"] <= t_index < buf["it1"]:
                    k = t_index - buf["it0"]
                    if buf["orientation"] == "horizontal":
                        self_Ez = self.Ez[buf["_slx"], buf["_y"]]
                        self_Hx = self.Hx[buf["_slx"], buf["_y"]]
                        self_Hy = self.Hy[buf["_slx"], buf["_y"]]
                    else:  # vertical
                        self_Ez = self.Ez[buf["_x"], buf["_sly"]]
                        self_Hx = self.Hx[buf["_x"], buf["_sly"]]
                        self_Hy = self.Hy[buf["_x"], buf["_sly"]]

                    buf["Ez"][k, :] = self_Ez
                    buf["Hx"][k, :] = self_Hx
                    buf["Hy"][k, :] = self_Hy

            # record
            if self.is_include_history and (t_index % self.record_stride) == 0:
                self.Hx_history[rec_idx, :, :] = self.Hx
                self.Hy_history[rec_idx, :, :] = self.Hy
                self.Ez_history[rec_idx, :, :] = self.Ez
                self.Dz_history[rec_idx, :, :] = self.Dz
                rec_idx += 1

        # --- finalize monitor outputs (drop private helper keys) ---
        self.monitor_results = []
        for buf in monitor_results:
            out = {k: v for k, v in buf.items() if not k.startswith("_")}
            self.monitor_results.append(out)

        # ---------- animation ----------

    def show_animation(self, fps=10, dynamic_clim=True, clim_smooth=0.2, pad=1e-12):
        """
        2x2: [ n-map , Hx ]
             [  Hy   , Ez ]
        Adds red markers/lines for sources and black translucent PML patches.
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        if not hasattr(self, "Hx_history") or self.Hx_history.size == 0:
            raise RuntimeError("No recorded field history. Run sim.run(...) first or load from file.")

        if not hasattr(self, "is_include_history") or self.is_include_history == False:
            raise RuntimeError("No recorded field history. Set sim.is_include_history=True and rerun the simulation.")

        Nx, Ny = self.Nx, self.Ny
        extent = [0, self.x_range, 0, self.y_range]
        mu_avg = 0.5 * (self.MRxx + self.MRyy)
        n_map = np.sqrt(self.ERzz * mu_avg)

        # global clim (fallback)
        if not dynamic_clim:
            vmax_Hx = np.max(np.abs(self.Hx_history)) + pad
            vmax_Hy = np.max(np.abs(self.Hy_history)) + pad
            vmax_Ez = np.max(np.abs(self.Ez_history)) + pad
            vmax_global = max(vmax_Hx, vmax_Hy, vmax_Ez)
            clim_H = (-vmax_global, vmax_global)
            clim_E = (-vmax_global, vmax_global)

        plt.ioff()
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        ax_n, ax_hx = axes[0]
        ax_hy, ax_ez = axes[1]

        # n-map
        im_n = ax_n.imshow(n_map.T, origin="lower", aspect="auto", extent=extent, cmap="viridis")
        im_n.set_clim(np.min(n_map), np.max(n_map))
        fig.colorbar(im_n, ax=ax_n).set_label("n")
        ax_n.set_title("Refractive index")
        ax_n.set_xlabel("x (m)")
        ax_n.set_ylabel("y (m)")

        # Hx / Hy / Ez
        im_hx = ax_hx.imshow(self.Hx_history[0].T, origin="lower", aspect="auto", extent=extent, cmap="jet")
        fig.colorbar(im_hx, ax=ax_hx).set_label("Hx")
        ax_hx.set_title("Hx")
        ax_hx.set_xlabel("x (m)")
        ax_hx.set_ylabel("y (m)")

        im_hy = ax_hy.imshow(self.Hy_history[0].T, origin="lower", aspect="auto", extent=extent, cmap="jet")
        fig.colorbar(im_hy, ax=ax_hy).set_label("Hy")
        ax_hy.set_title("Hy")
        ax_hy.set_xlabel("x (m)")
        ax_hy.set_ylabel("y (m)")

        im_ez = ax_ez.imshow(self.Ez_history[0].T, origin="lower", aspect="auto", extent=extent, cmap="jet")
        fig.colorbar(im_ez, ax=ax_ez).set_label("Ez")
        ax_ez.set_title("Ez")
        ax_ez.set_xlabel("x (m)")
        ax_ez.set_ylabel("y (m)")

        # time text
        time_text = ax_ez.text(0.02, 0.02, "", transform=ax_ez.transAxes,
                               bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

        # --- draw PML patches (black, alpha=0.3) ---
        def maybe_add_pml(ax):
            if self.pml_width is None: return
            # compute cell widths used when add_PML ran
            if isinstance(self.pml_width, int):
                npx = npy = max(1, int(self.pml_width))
            else:
                npx = max(1, int(np.ceil(float(self.pml_width) / self.dx)))
                npy = max(1, int(np.ceil(float(self.pml_width) / self.dy)))
            if self.pml_direction in ('xy', 'x'):
                xw = npx * self.dx
                ax.add_patch(Rectangle((0, 0), xw, self.y_range, facecolor='black', alpha=0.3, lw=0))
                ax.add_patch(
                    Rectangle((self.x_range - xw, 0), xw, self.y_range, facecolor='black', alpha=0.3, lw=0))
            if self.pml_direction in ('xy', 'y'):
                yw = npy * self.dy
                ax.add_patch(Rectangle((0, 0), self.x_range, yw, facecolor='black', alpha=0.3, lw=0))
                ax.add_patch(
                    Rectangle((0, self.y_range - yw), self.x_range, yw, facecolor='black', alpha=0.3, lw=0))

        for ax in (ax_n, ax_hx, ax_hy, ax_ez):
            maybe_add_pml(ax)

        # --- draw sources as red markers/lines ---
        def draw_sources(ax):
            for s in self.sources:
                # convert indices back to meters
                x0 = s["ix0"] * self.dx
                x1 = s["ix1"] * self.dx
                y0 = s["iy0"] * self.dy
                y1 = s["iy1"] * self.dy
                if s["kind"] == 'point':
                    ax.plot([x0], [y0], 'o', color='red', ms=5, mew=0)
                else:
                    # line-soft or sftf lines
                    if s["ix0"] != s["ix1"]:  # horizontal
                        ax.plot([x0, x1], [y0, y0], '-', color='red', lw=2)
                    else:  # vertical
                        ax.plot([x0, x0], [y0, y1], '-', color='red', lw=2)

        for ax in (ax_n, ax_hx, ax_hy, ax_ez):
            draw_sources(ax)

        # clims
        if dynamic_clim:
            frame0_max = max(np.max(np.abs(self.Hx_history[0])) + pad,
                             np.max(np.abs(self.Hy_history[0])) + pad,
                             np.max(np.abs(self.Ez_history[0])) + pad)
            smoothed_vmax = frame0_max
            im_hx.set_clim(-smoothed_vmax, smoothed_vmax)
            im_hy.set_clim(-smoothed_vmax, smoothed_vmax)
            im_ez.set_clim(-smoothed_vmax, smoothed_vmax)
        else:
            im_hx.set_clim(*clim_H)
            im_hy.set_clim(*clim_H)
            im_ez.set_clim(*clim_E)

        def _update(frame):
            nonlocal smoothed_vmax
            im_hx.set_data(self.Hx_history[frame].T)
            im_hy.set_data(self.Hy_history[frame].T)
            im_ez.set_data(self.Ez_history[frame].T)
            if dynamic_clim:
                vmax_now = max(np.max(np.abs(self.Hx_history[frame])) + pad,
                               np.max(np.abs(self.Hy_history[frame])) + pad,
                               np.max(np.abs(self.Ez_history[frame])) + pad)
                smoothed_vmax = (1.0 - clim_smooth) * vmax_now + clim_smooth * smoothed_vmax
                v = max(smoothed_vmax, pad)
                im_hx.set_clim(-v, v)
                im_hy.set_clim(-v, v)
                im_ez.set_clim(-v, v)
            tE = frame * getattr(self, "record_stride", 1) * self.dt
            time_text.set_text(f"t = {tE * 1e12:.3f} ps")
            return im_hx, im_hy, im_ez, time_text

        interval_ms = 1000.0 / max(1, fps)
        anim = FuncAnimation(fig, _update, frames=self.Nt_rec, interval=interval_ms, blit=True, repeat=False)
        plt.show()

    def NF2FF(self, top, bottom, left, right, freqs, nphi=361):
        """
        2D NF->FF (TMz / E-mode) using four line monitors (top, bottom, left, right).
        Each monitor index must refer to a line fully in free space (er=mr=1).

        Args
        ----
        top, bottom, left, right : int
            Indices into self.monitor_results for the four sides of the box.
            'top'  : y = y_high (horizontal, x from x1->x2)
            'bottom': y = y_low  (horizontal, x from x1->x2)
            'left' : x = x_low   (vertical,   y from y1->y2)
            'right': x = x_high  (vertical,   y from y1->y2)
        freqs : 1D array_like
            Frequencies (Hz) at which to compute the FF pattern (θ fixed to 90°, sweep φ).
        nphi : int
            Number of φ samples (0..2π). Default 361.

        Returns
        -------
        ff : dict with keys
            'phi'            : (nphi,) φ grid in radians
            'freqs'          : (Nf,)   frequencies (Hz)
            'Etheta'         : (Nf, nphi) complex
            'Ephi'           : (Nf, nphi) complex (zeros for 2D TMz)
            'Htheta'         : (Nf, nphi) complex (zeros for 2D TMz)
            'Hphi'           : (Nf, nphi) complex
            'Ptheta'         : (Nf, nphi) power density from Eθ   [|Eθ|^2/(2η0)]
            'Pphi'           : (Nf, nphi) (zeros for 2D TMz)
        """
        import numpy as np

        # --- helpers
        def _phasor_time_series(series, t, freqs):
            # tutorial DFT: Δt * Σ e^{-j2π f t} f(t)
            t = np.asarray(t, float)  # (T,)
            freqs = np.asarray(freqs, float)  # (Nf,)
            dt = self.dt
            # (Nf, T) kernel
            K = np.exp(-1j * 2 * np.pi * freqs[:, None] * t[None, :]) * dt
            # (T, L) → (Nf, L)
            return K @ series

        # --- get monitors
        M = self.monitor_results
        mT = M[int(top)]
        mB = M[int(bottom)]
        mL = M[int(left)]
        mR = M[int(right)]

        # --- sanity: orientations
        for midx, m, need in [(top, mT, "horizontal"),
                              (bottom, mB, "horizontal"),
                              (left, mL, "vertical"),
                              (right, mR, "vertical")]:
            ori = m.get("orientation", "").lower()
            if ori != need:
                raise ValueError(f"Monitor {midx} must be {need}, got '{ori}'.")

        # --- free-space checks (er=mr=1 on each line)
        def _check_free_space(m):
            if m["orientation"] == "horizontal":
                ix0, ix1, y = m["ix0"], m["ix1"], m["iy0"]
                er = self.ERzz[ix0:ix1, y]
                mx = self.MRxx[ix0:ix1, y]
                my = self.MRyy[ix0:ix1, y]
            else:
                x, iy0, iy1 = m["ix0"], m["iy0"], m["iy1"]
                er = self.ERzz[x, iy0:iy1]
                mx = self.MRxx[x, iy0:iy1]
                my = self.MRyy[x, iy0:iy1]
            if not (np.allclose(er, 1.0) and np.allclose(mx, 1.0) and np.allclose(my, 1.0)):
                raise ValueError("All four NF2FF monitors must lie entirely in free space (er=mr=1).")

        for m in (mT, mB, mL, mR):
            _check_free_space(m)

        # --- geometry / coordinates for phase factors
        # Positions of sample points on each line (center of Yee cell)
        # Top & bottom (horizontal)
        xT = np.arange(mT["ix0"], mT["ix1"]) * self.dx
        yT = np.full_like(xT, mT["iy0"] * self.dy, dtype=float)
        xB = np.arange(mB["ix0"], mB["ix1"]) * self.dx
        yB = np.full_like(xB, mB["iy0"] * self.dy, dtype=float)
        # Right & left (vertical)
        yR = np.arange(mR["iy0"], mR["iy1"]) * self.dy
        xR = np.full_like(yR, mR["ix0"] * self.dx, dtype=float)
        yL = np.arange(mL["iy0"], mL["iy1"]) * self.dy
        xL = np.full_like(yL, mL["ix0"] * self.dx, dtype=float)

        # Differential lengths for integration
        dx = self.dx
        dy = self.dy

        # --- build time grids per monitor
        tT = np.arange(mT["it0"], mT["it1"]) * self.dt
        tB = np.arange(mB["it0"], mB["it1"]) * self.dt
        tL = np.arange(mL["it0"], mL["it1"]) * self.dt
        tR = np.arange(mR["it0"], mR["it1"]) * self.dt

        # --- get time series arrays (T,L) for each side
        EzT, HxT, HyT = mT["Ez"], mT["Hx"], mT["Hy"]
        EzB, HxB, HyB = mB["Ez"], mB["Hx"], mB["Hy"]
        EzL, HxL, HyL = mL["Ez"], mL["Hx"], mL["Hy"]
        EzR, HxR, HyR = mR["Ez"], mR["Hx"], mR["Hy"]

        # --- phasors at requested freqs: (Nf, L)
        freqs = np.asarray(freqs, float)
        ET = _phasor_time_series(EzT, tT, freqs) * self.eta0
        EB = _phasor_time_series(EzB, tB, freqs) * self.eta0
        EL = _phasor_time_series(EzL, tL, freqs) * self.eta0
        ER = _phasor_time_series(EzR, tR, freqs) * self.eta0

        HxT_f = _phasor_time_series(HxT, tT, freqs)
        HxB_f = _phasor_time_series(HxB, tB, freqs)
        HxL_f = _phasor_time_series(HxL, tL, freqs)
        HxR_f = _phasor_time_series(HxR, tR, freqs)

        HyT_f = _phasor_time_series(HyT, tT, freqs)
        HyB_f = _phasor_time_series(HyB, tB, freqs)
        HyL_f = _phasor_time_series(HyL, tL, freqs)
        HyR_f = _phasor_time_series(HyR, tR, freqs)

        # --- φ grid (θ = 90° plane); r-hat = (cosφ, sinφ)
        phi = np.linspace(0.0, 2 * np.pi, int(nphi), endpoint=False)
        cφ = np.cos(phi)[None, :]  # (1, nphi)
        sφ = np.sin(phi)[None, :]

        # --- phase factors e^{-jk rhat·r'} for each side (Nf,nφ,L)
        k0 = 2 * np.pi * freqs[:, None] / self.c0  # (Nf,1)

        def _phase_x(xline, yconst):
            # rhat·r' = x cosφ + y sinφ
            return np.exp(-1j * (k0[..., None]) * (xline[None, None, :] * cφ[..., None] +
                                                   yconst[None, None, :] * sφ[..., None]))

        def _phase_y(xconst, yline):
            return np.exp(-1j * (k0[..., None]) * (xconst[None, None, :] * cφ[..., None] +
                                                   yline[None, None, :] * sφ[..., None]))

        PH_T = _phase_x(xT, yT)  # top    (Nf,nφ,LT)
        PH_B = _phase_x(xB, yB)  # bottom (Nf,nφ,LB)
        PH_R = _phase_y(xR, yR)  # right  (Nf,nφ,LR)
        PH_L = _phase_y(xL, yL)  # left   (Nf,nφ,LL)

        # Nθ(φ) =  - ∫ Hx_bottom e^{-jk·r'} dx - ∫ Hy_right e^{-jk·r'} dy
        #           + ∫ Hx_top    e^{-jk·r'} dx + ∫ Hy_left  e^{-jk·r'} dy
        # Lφ(φ) =  - sinφ ∫ Ez_bottom e^{-jk·r'} dx + cosφ ∫ Ez_right e^{-jk·r'} dy
        #           + sinφ ∫ Ez_top    e^{-jk·r'} dx - cosφ ∫ Ez_left  e^{-jk·r'} dy
        #
        # Discretize: sums over samples with dℓ=dx or dy.
        # Shapes:
        #   Hx*_f : (Nf, Lx),  PH_* : (Nf, nφ, Lx)

        def _int_x(Fx, PH):  # integrate along x with dx
            return np.sum(Fx[:, None, :] * PH, axis=2) * dx  # (Nf,nφ)

        def _int_y(Fy, PH):  # integrate along y with dy
            return np.sum(Fy[:, None, :] * PH, axis=2) * dy

        Nθ = (- _int_x(HxB_f, PH_B)
              - _int_y(HyR_f, PH_R)
              + _int_x(HxT_f, PH_T)
              + _int_y(HyL_f, PH_L))  # (Nf,nφ)

        Lφ = (- sφ * _int_x(EB, PH_B)
              + cφ * _int_y(ER, PH_R)
              + sφ * _int_x(ET, PH_T)
              - cφ * _int_y(EL, PH_L))  # (Nf,nφ)

        # --- Far fields (θ=90°). From slide:
        # Eθ = j k e^{jk r} / (4π r) ( η Nθ + Lφ )
        # Hφ = j k e^{jk r} / (4π r) ( Lφ/η + Nθ )
        # We omit the common scalar prefactor (j k e^{jkr}/(4πr)) because pattern

        eta0 = self.eta0
        Eθ = eta0 * Nθ + Lφ
        Hφ = (Lφ / eta0) + Nθ

        # zeros for the orthogonal components in 2D TMz
        Z = np.zeros_like(Eθ)
        Eφ = Z.copy()
        Hθ = Z.copy()

        # Power densities (per polarization)
        Pθ = (np.abs(Eθ) ** 2) / (2.0 * eta0)
        Pφ = (np.abs(Eφ) ** 2) / (2.0 * eta0)  # zero

        ff = dict(
            phi=phi, freqs=freqs,
            Etheta=Eθ, Ephi=Eφ,
            Htheta=Hθ, Hphi=Hφ,
            Ptheta=Pθ, Pphi=Pφ
        )
        return ff

    def show_FF(self, ff, freq_idx=0, component="Etheta", db=True, dr_db=40, normalize="max"):
        """
        Polar plot of a chosen far-field component vs φ, with one or many frequency indices.

        Args
        ----
        ff : dict
            Output from NF2FF() with keys: 'phi','freqs', and the field/power arrays.
        freq_idx : int or sequence of ints
            Which frequency index/indices to plot. e.g. 0, -1, [0, 10, 20]
        component : {"Etheta","Ephi","Htheta","Hphi","Ptheta","Pphi"}
            Which quantity to plot on the polar axis.
        db : bool
            If True plot in dB (normalized); else plot linear magnitude (normalized if requested).
        dr_db : float
            Dynamic range floor in dB when db=True (e.g., 40 → floor at −40 dB).
        normalize : {"max","integral",None}
            Per-curve normalization before plotting:
              - "max"      : divide by max(|y|) over φ for that curve
              - "integral" : divide by sqrt(mean(|y|^2)) over φ (RMS)
              - None       : no normalization
        """
        import numpy as np
        import matplotlib.pyplot as plt

        # --- inputs and validation
        phi = np.asarray(ff["phi"])  # (nphi,)
        freqs = np.asarray(ff["freqs"])  # (Nf,)
        key = str(component)
        if key not in ("Etheta", "Ephi", "Htheta", "Hphi", "Ptheta", "Pphi"):
            raise ValueError("component must be one of: Etheta, Ephi, Htheta, Hphi, Ptheta, Pphi")
        if key not in ff:
            raise KeyError(f"{key} not found in ff dict")

        data = np.asarray(ff[key])  # (Nf, nphi)
        if data.ndim != 2 or data.shape[1] != phi.size:
            raise ValueError(f"ff['{key}'] has shape {data.shape}, expected (Nf, nphi={phi.size}).")

        # normalize freq_idx to a list
        if isinstance(freq_idx, (list, tuple, np.ndarray)):
            idx_list = [int(i) for i in freq_idx]
        else:
            idx_list = [int(freq_idx)]
        # clip/validate indices and build labels
        Nf = data.shape[0]
        for i in idx_list:
            if not (-Nf <= i < Nf):
                raise IndexError(f"freq_idx {i} out of range for Nf={Nf}")

        # Decide dB rule: fields use 20*log10, powers use 10*log10
        is_power = key.startswith("P")
        log_factor = 10.0 if is_power else 20.0

        # Per-curve normalization helper
        def _norm_curve(y):
            y = np.asarray(y)
            if normalize == "max":
                s = np.max(np.abs(y)) or 1.0
            elif normalize == "integral":
                s = np.sqrt(np.mean(np.abs(y) ** 2)) or 1.0
            else:
                s = 1.0
            return y / s

        # dB conversion with floor
        def _to_db(y):
            y = np.abs(y)
            # Always normalize to own max before dB floor so curves share 0 dB reference
            y = y / (np.max(y) or 1.0)
            return log_factor * np.log10(np.maximum(y, 10 ** (-dr_db / log_factor)))

        # --- figure: single polar plot
        fig = plt.figure(figsize=(6.8, 5.4))
        ax = plt.subplot(1, 1, 1, projection='polar')
        ax.grid(True, alpha=0.3)
        ax.set_theta_zero_location("E")  # 0° at +x (East)
        ax.set_theta_direction(1)  # CCW

        # plot each requested frequency slice
        handles = []
        labels = []
        for i in idx_list:
            i_eff = i % Nf
            y = data[i_eff, :]  # (nphi,)
            y = _norm_curve(y)  # apply chosen normalization

            if db:
                r = _to_db(y)
                ax.set_rlim(-dr_db, 0)  # fixed dB range
                rlabel = "dB (normalized)"
            else:
                r = np.abs(y)
                # If normalized, fix to [0,1]; else autoscale
                if normalize:
                    ax.set_rlim(0, 1.0)
                rlabel = "Magnitude (normalized)" if normalize else "Magnitude"

            h, = ax.plot(phi, r, lw=1, ls='-')
            handles.append(h)
            labels.append(f"{freqs[i_eff] / 1e9:.3f} GHz")

        ax.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.2, 1.10))
        ax.set_rlabel_position(135)

        # Title: what we plotted + "(φ)"
        ax.set_title(f"{key} (φ)")

        # y-axis label text (note: polar axes don't have a natural y-label; place as annotation)
        ax.annotate(rlabel, xy=(0.98, 0.02), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

        fig.tight_layout()
        plt.show()

    # ---------- state dict / I/O ----------

    def to_state_dict(self, dtype=np.float32, include_histories=True):
        def cast(a):
            return a.astype(dtype, copy=False) if isinstance(a, np.ndarray) and np.issubdtype(a.dtype,
                                                                                              np.floating) else a

        # ---- sources table: [kind_code, ix0, ix1, iy0, iy1, amp, t0, tw, fmin, fmax]
        kind_code_map = {'point': 0, 'line-soft': 1, 'sftf-x': 2, 'sftf-y': 3, 'waveguide-y': 4}
        src_mat = np.zeros((len(self.sources), 10), dtype=np.float64)
        for i, s in enumerate(self.sources):
            src_mat[i, 0] = kind_code_map.get(s["kind"], 0)
            src_mat[i, 1] = s["ix0"]
            src_mat[i, 2] = s["ix1"]
            src_mat[i, 3] = s["iy0"]
            src_mat[i, 4] = s["iy1"]
            src_mat[i, 5] = s["amplitude"]
            src_mat[i, 6] = s["t0"]
            src_mat[i, 7] = s["tw"]
            src_mat[i, 8] = (-1.0 if s["f_min"] is None else float(s["f_min"]))
            src_mat[i, 9] = s["f_max"]

        # ---- waveguide-y blobs (optional, only for those sources that have profiles)
        wg_meta = []  # rows: [src_index, offset, length, n_eff]
        ez_blob = []
        hx_blob = []
        offset = 0
        for idx, s in enumerate(self.sources):
            if s.get("kind") == "waveguide-y" and ("Ez_src" in s) and ("Hx_src" in s):
                ez = np.asarray(s["Ez_src"], dtype=dtype).ravel()
                hx = np.asarray(s["Hx_src"], dtype=dtype).ravel()
                L = int(min(len(ez), len(hx)))
                if L > 0:
                    ez_blob.append(ez[:L])
                    hx_blob.append(hx[:L])
                    neff = float(s.get("n_eff", np.nan))
                    wg_meta.append([idx, offset, L, neff])
                    offset += L
        wg_meta = np.asarray(wg_meta, dtype=np.float64)
        wg_ez = (np.concatenate(ez_blob).astype(dtype) if len(ez_blob) else np.array([], dtype=dtype))
        wg_hx = (np.concatenate(hx_blob).astype(dtype) if len(hx_blob) else np.array([], dtype=dtype))

        # ---- periodic flags as a tiny array of strings (e.g., ['x','y'])
        periodic_arr = np.array(self.periodic, dtype='<U1')

        state = {
            # grid/time
            "x_range": float(self.x_range), "y_range": float(self.y_range),
            "Nx": int(self.Nx), "Ny": int(self.Ny),
            "dx": float(self.dx), "dy": float(self.dy),
            "Nt": int(self.Nt), "dt": float(self.dt),
            "f_max": float(self.f_max),
            "f_min": (-1.0 if self.f_min is None else float(self.f_min)),

            # recording
            "record_stride": int(getattr(self, "record_stride", 1)),
            "Nt_rec": int(getattr(self, "Nt_rec", 0)),

            # materials & losses
            "ERzz": cast(self.ERzz), "MRxx": cast(self.MRxx), "MRyy": cast(self.MRyy),
            "sigma_hx": cast(self.sigma_hx), "sigma_hy": cast(self.sigma_hy),
            "sigma_dx": cast(self.sigma_dx), "sigma_dy": cast(self.sigma_dy),

            # PML params
            "pml_width_val": (float(self.pml_width) if isinstance(self.pml_width, (int, float)) else -1.0),
            "pml_width_kind": (
                0 if isinstance(self.pml_width, int) else (1 if isinstance(self.pml_width, float) else -1)),
            "sigma_max": (float(self.sigma_max) if self.sigma_max is not None else -1.0),
            "pml_order": int(self.pml_order) if self.pml_order is not None else -1,
            "pml_direction": np.array(self.pml_direction),

            # periodic
            "periodic": periodic_arr,

            # sources
            "sources": src_mat,
            "avg_freqs": np.array(self.avg_freqs, dtype=np.float64),

            # waveguide-y extras
            "wg_meta": wg_meta,  # shape (Nw, 4) columns: [src_index, offset, length, n_eff]
            "wg_Ez_blob": wg_ez,  # concatenated Ez_src for all waveguide-y sources
            "wg_Hx_blob": wg_hx,  # concatenated Hx_src for all waveguide-y sources

            # field histories (optional)
            "Hx_history": cast(getattr(self, "Hx_history", np.array([]))),
            "Hy_history": cast(getattr(self, "Hy_history", np.array([]))),
            "Ez_history": cast(getattr(self, "Ez_history", np.array([]))),
            "Dz_history": cast(getattr(self, "Dz_history", np.array([]))),
        }

        if not include_histories:
            state["Hx_history"] = np.array([])
            state["Hy_history"] = np.array([])
            state["Ez_history"] = np.array([])
            state["Dz_history"] = np.array([])

        return state

    def save_npz(self, path, dtype=np.float32, include_histories=True):
        import os, tempfile
        state = self.to_state_dict(dtype=dtype, include_histories=include_histories)
        d = os.path.dirname(os.path.abspath(path)) or "."
        fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".tmp.", dir=d)
        os.close(fd)
        try:
            np.savez_compressed(tmp, **state)
            os.replace(tmp, path)
        except Exception:
            try:
                os.remove(tmp)
            except Exception:
                pass
            raise

    @classmethod
    def load_npz(cls, path, mmap=False):
        data = np.load(path, allow_pickle=False, mmap_mode=("r" if mmap else None))

        # core sim
        sim = cls(
            x_range=float(data["x_range"]),
            y_range=float(data["y_range"]),
            Nx=int(data["Nx"]), Ny=int(data["Ny"]),
            f_max=float(data["f_max"]) if "f_max" in data.files else 1.0,
            Nt=int(data["Nt"]), dt=float(data["dt"]),
        )
        sim.dx = float(data["dx"])
        sim.dy = float(data["dy"])
        sim.record_stride = int(data["record_stride"]) if "record_stride" in data.files else 1
        sim.Nt_rec = int(data["Nt_rec"]) if "Nt_rec" in data.files else 0
        sim.f_min = None if ("f_min" not in data.files or float(data["f_min"]) < 0) else float(data["f_min"])

        # materials & loss
        sim.ERzz = data["ERzz"]
        sim.MRxx = data["MRxx"]
        sim.MRyy = data["MRyy"]
        sim.sigma_hx = data["sigma_hx"]
        sim.sigma_hy = data["sigma_hy"]
        sim.sigma_dx = data["sigma_dx"]
        sim.sigma_dy = data["sigma_dy"]

        # PML params
        sim.pml_width = None
        if "pml_width_kind" in data.files and "pml_width_val" in data.files:
            kind = int(data["pml_width_kind"])
            val = float(data["pml_width_val"])
            if kind == 0:
                sim.pml_width = int(val)
            elif kind == 1:
                sim.pml_width = float(val)
        sim.sigma_max = None if "sigma_max" not in data.files else (
            None if float(data["sigma_max"]) < 0 else float(data["sigma_max"]))
        sim.pml_order = int(data["pml_order"]) if "pml_order" in data.files else 3
        sim.pml_direction = str(np.array(data["pml_direction"])) if "pml_direction" in data.files else 'xy'

        # periodic flags
        sim.periodic = list(np.array(data["periodic"])) if "periodic" in data.files else []

        # sources
        sim.sources = []
        sim.avg_freqs = list(np.array(data["avg_freqs"], dtype=float)) if "avg_freqs" in data.files else []
        if "sources" in data.files:
            mat = np.array(data["sources"])
            kind_map = {0: 'point', 1: 'line-soft', 2: 'sftf-x', 3: 'sftf-y', 4: 'waveguide-y'}
            for row in mat:
                s = dict(
                    kind=kind_map.get(int(row[0]), 'point'),
                    ix0=int(row[1]), ix1=int(row[2]),
                    iy0=int(row[3]), iy1=int(row[4]),
                    amplitude=float(row[5]), t0=float(row[6]), tw=float(row[7]),
                    f_min=(None if float(row[8]) < 0 else float(row[8])),
                    f_max=float(row[9]),
                )
                sim.sources.append(s)

        # waveguide-y extras (optional; if missing, run will still work — the mode will be recomputed when you re-add)
        if "wg_meta" in data.files and data["wg_meta"].size:
            meta = np.array(data["wg_meta"])
            ez_blob = np.array(data["wg_Ez_blob"])
            hx_blob = np.array(data["wg_Hx_blob"])
            for src_index, offset, length, n_eff in meta.astype(int):
                off = int(offset)
                L = int(length)
                s = sim.sources[int(src_index)]
                s["Ez_src"] = ez_blob[off:off + L]
                s["Hx_src"] = hx_blob[off:off + L]
                s["n_eff"] = float(n_eff)

        # field histories (optional)
        for name in ("Hx_history", "Hy_history", "Ez_history", "Dz_history"):
            if name in data.files:
                setattr(sim, name, data[name])

        return sim

    @classmethod
    def animate_npz(cls, path, fps=60, dynamic_clim=True, clim_smooth=0.2, mmap=False):
        sim = cls.load_npz(path, mmap=mmap)
        sim.show_animation(fps=fps, dynamic_clim=dynamic_clim, clim_smooth=clim_smooth)
        return sim
