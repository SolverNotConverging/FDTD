import numpy as np
from matplotlib.patches import Rectangle
from tqdm import tqdm


class FDTD_2D_Hz:
    """
    2D TEz FDTD with:
      • multiple sources (point, line-soft, SF/TF, modal waveguide ports)
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
        self.ERxx = np.ones((self.Nx, self.Ny))
        self.ERyy = np.ones((self.Nx, self.Ny))
        self.MRzz = np.ones((self.Nx, self.Ny))

        # fields, we use normalized E' D' H' B'

        # E'=E, D'=eps0*D, D'=eps_r*E
        self.Dx = np.zeros((self.Nx, self.Ny))
        self.Ex = np.zeros((self.Nx, self.Ny))

        self.Dy = np.zeros((self.Nx, self.Ny))
        self.Ey = np.zeros((self.Nx, self.Ny))

        # H' = sqrt(mu0/eps0)*H, B'=sqrt(eps0*mu0)*B, B'=mu_r*H'
        self.Bz = np.zeros((self.Nx, self.Ny))
        self.Hz = np.zeros((self.Nx, self.Ny))

        # PML coefficients
        self.pml_width = None
        self.sigma_max = None
        self.pml_order = 3
        self.pml_direction = 'xy'  # 'x'|'y'|'xy'

        self.M = self.c0 * self.dt

        self.kappa_x = np.ones((self.Nx, self.Ny))
        self.kappa_y = np.ones((self.Nx, self.Ny))

        self.alpha_x = np.zeros((self.Nx, self.Ny))
        self.alpha_y = np.zeros((self.Nx, self.Ny))

        self.sigma_x = np.zeros((self.Nx, self.Ny))
        self.sigma_y = np.zeros((self.Nx, self.Ny))

        self.b_Dx_y = np.zeros((self.Nx, self.Ny))
        self.b_Dy_x = np.zeros((self.Nx, self.Ny))
        self.b_Bz_x = np.zeros((self.Nx, self.Ny))
        self.b_Bz_y = np.zeros((self.Nx, self.Ny))

        self.c_Dy_x = np.zeros((self.Nx, self.Ny))
        self.c_Dx_y = np.zeros((self.Nx, self.Ny))
        self.c_Bz_x = np.zeros((self.Nx, self.Ny))
        self.c_Bz_y = np.zeros((self.Nx, self.Ny))

        # update coefficients
        self.Psi_Dx_y = np.zeros((self.Nx, self.Ny))
        self.Psi_Dy_x = np.zeros((self.Nx, self.Ny))
        self.Psi_Bz_x = np.zeros((self.Nx, self.Ny))
        self.Psi_Bz_y = np.zeros((self.Nx, self.Ny))

        # curls + integrals
        self.d_Ey_x = np.zeros((self.Nx, self.Ny))
        self.d_Ex_y = np.zeros((self.Nx, self.Ny))
        self.d_Hz_x = np.zeros((self.Nx, self.Ny))
        self.d_Hz_y = np.zeros((self.Nx, self.Ny))

        # multi-source list ---
        # each source is a dict with keys:
        #   kind: 'point' | 'line-soft' | 'sftf' | 'waveguide-x' | 'waveguide-y'
        #   ix0, ix1, iy0, iy1 (ints; for points, ix0,iy0 used; for spans, [ix0,ix1), [iy0,iy1))
        #   amplitude, t0, tw, f_min (or None), f_max
        #   (for 'sftf'):
        #       angle      : propagation angle θ (radians) measured from +x toward +y
        #       f0         : center frequency used for kx, ky
        #       kx, ky, k0 : components and magnitude of k-vector
        #       delay_xlo, delay_xhi : 1D arrays of time delay for left/right TF/SF edges
        #       delay_ylo, delay_yhi : 1D arrays of time delay for bottom/top TF/SF edges
        #       ix_lo, ix_hi, iy_lo, iy_hi : sorted interior TF region indices
        self.sources = []
        self.avg_freqs = []  # one per source (spectral centroid for info/diagnostics)

        # field monitor dictionary
        self.monitors = []
        self.monitor_results = []

    # ---------- geometry helpers ----------
    def add_rectangle(self, ER, MR, x_position, y_position):
        if isinstance(ER, (list, tuple, np.ndarray)) and len(ER) == 3:
            ERxx_obj = float(ER[0])
            ERyy_obj = float(ER[1])
        else:
            ERxx_obj = float(ER)
            ERyy_obj = float(ER)

        if isinstance(MR, (list, tuple, np.ndarray)) and len(MR) == 3:
            MRzz_obj = float(MR[2])
        else:
            MRzz_obj = float(MR)

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
                self.ERxx[i, j] = (1.0 - f) * self.ERxx[i, j] + f * ERxx_obj
                self.ERyy[i, j] = (1.0 - f) * self.ERyy[i, j] + f * ERyy_obj
                self.MRzz[i, j] = (1.0 - f) * self.MRzz[i, j] + f * MRzz_obj

    def add_circle(self, ER, MR, center, radius, nsub=6):
        """
        Add a (possibly anisotropic) circular object with subpixel edge smoothing.

        ER: float (isotropic) or (ERxx, ERyy, ERzz). TEz uses ERxx, ERyy.
        MR: float (isotropic) or (MRxx, MRyy, MRzz). TEz uses MRzz.
        center: (cx, cy) where each element is int (edge index) or float (meters).
        radius: float, meters.
        nsub: supersamples per axis (nsub x nsub per cell) for area fraction.
        """
        # --- parse materials (TMz: Ez/Dz along z, Hx/Hy in-plane) ---
        if isinstance(ER, (list, tuple, np.ndarray)) and len(ER) == 3:
            ERxx_obj = float(ER[0])
            ERyy_obj = float(ER[1])
        else:
            ERxx_obj = float(ER)
            ERyy_obj = float(ER)

        if isinstance(MR, (list, tuple, np.ndarray)) and len(MR) == 3:
            MRzz_obj = float(ER[2])
        else:
            MRzz_obj = float(ER)

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

                self.ERxx[i, j] = (1.0 - f) * self.ERxx[i, j] + f * ERxx_obj
                self.ERyy[i, j] = (1.0 - f) * self.ERyy[i, j] + f * ERyy_obj
                self.MRzz[i, j] = (1.0 - f) * self.MRzz[i, j] + f * MRzz_obj

        # ---------- PML ----------

    def add_PML(self, pml_width, order=3, direction='xy', sigma_max=None,
                kappa_max=7, alpha_max=0.025, R0=1e-8):

        self.pml_width = int(pml_width) if isinstance(pml_width, int) else float(pml_width)
        self.pml_order = int(order)
        self.pml_direction = str(direction)

        # convert to #cells in each axis
        if isinstance(pml_width, int):
            npx = npy = max(1, int(pml_width))
        elif isinstance(pml_width, float):
            npx = max(1, int(np.ceil(pml_width / self.dx)))
            npy = max(1, int(np.ceil(pml_width / self.dy)))
        else:
            raise TypeError("pml_width must be int (cells) or float (meters).")

        # physical thickness L for the formula
        Lx = npx * self.dx if npx > 0 else np.inf
        Ly = npy * self.dy if npy > 0 else np.inf
        L = min(Lx, Ly)

        if sigma_max is None:
            # slide-11: sigma_max = -(n+1) * log10(R0) / (2 * eta0 * L)
            self.sigma_max = -(self.pml_order + 1) * np.log10(R0) / (2 * self.eta0 * L)
        else:
            self.sigma_max = float(sigma_max)

        if 'x' in direction:
            for i in range(npx):
                s = self.sigma_max * ((npx - i) / npx) ** order
                self.sigma_x[i, :] = s
                self.sigma_x[-i - 1, :] = s

                k = 1 + (kappa_max - 1) * ((npx - i) / npx) ** order
                self.kappa_x[i, :] = k
                self.kappa_x[-i - 1, :] = k

                a = alpha_max * (i / npx) ** order
                self.alpha_x[i, :] = a
                self.alpha_x[-i - 1, :] = a

        if 'y' in direction:
            for i in range(npy):
                s = self.sigma_max * ((npy - i) / npy) ** order
                self.sigma_y[:, i] = s
                self.sigma_y[:, -i - 1] = s

                k = 1 + (kappa_max - 1) * ((npy - i) / npy) ** order
                self.kappa_y[:, i] = k
                self.kappa_y[:, -i - 1] = k

                a = alpha_max * (i / npy) ** order
                self.alpha_y[:, i] = a
                self.alpha_y[:, -i - 1] = a

        if not ('x' in direction or 'y' in direction):
            raise TypeError("direction must be 'x', 'y', or 'xy'.")

    def _init_Coeff(self):

        # b-coeffs
        ex1 = self.sigma_x / (self.eps0 * self.kappa_x) + self.alpha_x / self.eps0
        self.b_Dy_x = np.exp(-ex1 * self.dt)

        ex2 = self.sigma_y / (self.eps0 * self.kappa_y) + self.alpha_y / self.eps0
        self.b_Dx_y = np.exp(-ex2 * self.dt)

        # set b=1 exactly outside PML
        mask_x0 = (self.sigma_x == 0) & (self.alpha_x == 0)
        mask_y0 = (self.sigma_y == 0) & (self.alpha_y == 0)
        self.b_Dy_x[mask_x0] = 1.0
        self.b_Dx_y[mask_y0] = 1.0

        # c-coeffs with safe division; c=0 where sigma=alpha=0
        den1 = self.sigma_x * self.kappa_x + self.alpha_x * self.kappa_x ** 2
        den2 = self.sigma_y * self.kappa_y + self.alpha_y * self.kappa_y ** 2

        self.c_By_x = np.zeros_like(self.sigma_x)
        self.c_Bx_y = np.zeros_like(self.sigma_y)

        good1 = den1 != 0
        good2 = den2 != 0
        self.c_Dy_x[good1] = (self.sigma_x[good1] / den1[good1]) * (self.b_Dy_x[good1] - 1.0)
        self.c_Dx_y[good2] = (self.sigma_y[good2] / den2[good2]) * (self.b_Dx_y[good2] - 1.0)

        # D uses the same b/c as B in TEz
        self.b_Bz_x, self.b_Bz_y = self.b_Dy_x, self.b_Dx_y
        self.c_Bz_x, self.c_Bz_y = self.c_Dy_x, self.c_Dx_y

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
        MRzz_vec = np.asarray(self.MRzz[lo:hi, iy], dtype=float)
        ERxx_vec = np.asarray(self.ERxx[lo:hi, iy], dtype=float)
        ERyy_vec = np.asarray(self.ERyy[lo:hi, iy], dtype=float)

        # sparse diagonals (NOTE: shape must be a tuple)
        MRzz_diag = spdiags(MRzz_vec, 0, shape=(Nx, Nx))
        ERxx_inv = spdiags(1.0 / ERxx_vec, 0, shape=(Nx, Nx))
        ERyy_inv = spdiags(1.0 / ERyy_vec, 0, shape=(Nx, Nx))

        # your difference operators (exact layout and scaling)
        d_plus = np.ones(Nx)
        d_minus = -np.ones(Nx)
        # DEX: offsets [1, 0] / (dx*k0)  -> upper diag = +1, main = -1
        DEX = spdiags([d_plus, d_minus], [1, 0], shape=(Nx, Nx)) / (dx * k0)
        # DHX: offsets [0, 1] / (dx*k0)  -> main = +1, lower = -1
        DHX = spdiags([d_plus, d_minus], [0, -1], shape=(Nx, Nx)) / (dx * k0)

        # operators
        A = MRzz_diag + DEX @ (ERxx_inv @ DHX)
        B = ERyy_inv

        # shift guess near n_core^2
        if guess is None:
            n_slice = np.sqrt(MRzz_vec * 0.5 * (ERxx_vec + ERyy_vec))
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
        Hz_modes = []
        Ex_modes = []

        for m in range(evecs.shape[1]):
            Hz = evecs[:, m]

            kmax = np.argmax(np.abs(Hz))
            phase = np.angle(Hz[kmax])
            Hz = (Hz * np.exp(-1j * phase)).real

            # Tutorial relation: h_x = - n_eff * mu_xx^{-1} * e_z
            Ex = -(n_eff[m]) * (ERxx_inv @ Hz)

            # MRxx_inv is a sparse diagonal; ensure dense vector
            Ex = Ex.A.squeeze() if hasattr(Ex, "A") else np.asarray(Ex).squeeze()
            Ex = Ex.real

            # Normalize Ex to max|.| = amplitude
            norm = amplitude / np.max(np.abs(Ex))
            Ex = Ex * norm
            Hz = Hz * norm

            Ex_modes.append(Ex)
            Hz_modes.append(Hz)

        Ex_modes = np.asarray(Ex_modes)
        Hz_modes = np.asarray(Hz_modes)

        return np.asarray(Hz_modes), np.asarray(Ex_modes), np.asarray(n_eff, dtype=float)

    def _wg_modes_x(self, iy0, iy1, ix, f_center, num_modes=4, guess=None, amplitude=1.0):
        import numpy as np
        from scipy.sparse import diags as spdiags
        from scipy.sparse.linalg import eigs

        lo, hi = (min(iy0, iy1), max(iy0, iy1))
        Ny = int(max(1, hi - lo))
        if Ny < 2:
            raise ValueError("waveguide-x: y-span too small; need at least 2 cells.")

        k0 = 2.0 * np.pi * float(f_center) / self.c0
        dy = self.dy

        MRzz_vec = np.asarray(self.MRzz[ix, lo:hi], dtype=float)
        ERxx_vec = np.asarray(self.ERxx[ix, lo:hi], dtype=float)
        ERyy_vec = np.asarray(self.ERyy[ix, lo:hi], dtype=float)

        MRzz_diag = spdiags(MRzz_vec, 0, shape=(Ny, Ny))
        ERxx_inv = spdiags(1.0 / ERxx_vec, 0, shape=(Ny, Ny))
        ERyy_inv = spdiags(1.0 / ERyy_vec, 0, shape=(Ny, Ny))

        d_plus = np.ones(Ny)
        d_minus = -np.ones(Ny)
        DEY = spdiags([d_plus, d_minus], [1, 0], shape=(Ny, Ny)) / (dy * k0)
        DHY = spdiags([d_plus, d_minus], [0, -1], shape=(Ny, Ny)) / (dy * k0)

        A = MRzz_diag + DHY @ (ERyy_inv @ DEY)
        B = ERxx_inv

        if guess is None:
            n_slice = np.sqrt(MRzz_vec * 0.5 * (ERxx_vec + ERyy_vec))
            n_guess = float(np.max(n_slice))
            guess = max(n_guess ** 2, 1.0)

        k = int(max(1, num_modes))
        evals, evecs = eigs(A, M=B, k=k, sigma=guess)

        n_eff = np.sqrt(np.maximum(evals.real, 0.0))
        order = np.argsort(-n_eff)
        evecs = evecs[:, order]
        n_eff = n_eff[order]

        Hz_modes = []
        Ey_modes = []

        for m in range(evecs.shape[1]):
            Hz = evecs[:, m]

            kmax = np.argmax(np.abs(Hz))
            phase = np.angle(Hz[kmax])
            Hz = (Hz * np.exp(-1j * phase)).real

            Ey = -(n_eff[m]) * (ERyy_inv @ Hz)
            Ey = Ey.A.squeeze() if hasattr(Ey, "A") else np.asarray(Ey).squeeze()
            Ey = Ey.real

            norm = amplitude / np.max(np.abs(Ey))

            Ey = Ey * norm
            Hz = Hz * norm

            Hz_modes.append(Hz)
            Ey_modes.append(Ey)

        Hz_modes = np.asarray(Hz_modes)
        Ey_modes = np.asarray(Ey_modes)

        return np.asarray(Hz_modes), np.asarray(Ey_modes), np.asarray(n_eff, dtype=float)

    # ---------- public API: add_source ----------
    def add_source(self, kind, x, y, amplitude=1.0, t0=None, tw=None,
                   f_min=None, f_max=None, mode_index=1,
                   modes_to_show=4, eig_guess=None, is_show=True,
                   angle=None):

        """
        Add a source.

        kind:
          'point'       : soft point into Bz at (x,y)
          'line-soft'   : soft line into Bz; give x=(ix0,ix1) & y=j or y=(j0,j1) & x=i
          'sftf'        : TF/SF rectangular box; give x=(x_lo,x_hi), y=(y_lo,y_hi) and angle
          'waveguide-x' : modal source on a vertical slice injecting toward +x
          'waveguide-y' : modal source on a horizontal slice injecting toward +y

        x, y:
          Either ints (indices) or floats (meters). For spans, pass (start, end).

        For 'sftf':
          - x must be a span (x_lo, x_hi) and y must be a span (y_lo, y_hi).
          - angle is the propagation angle θ in radians measured from +x toward +y.
        """

        k = kind.lower()
        if k not in ('point', 'line-soft', 'sftf', 'waveguide-x', 'waveguide-y'):
            raise ValueError("kind must be 'point', 'line-soft', 'sftf', 'waveguide-x', 'waveguide-y'.")

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

        # --- extra parameters for TF/SF angled source ('sftf') ---
        if k == 'sftf':
            # Require both x and y to be spans (non-zero length)
            if ix0 == ix1 or iy0 == iy1:
                raise ValueError("For 'sftf', x and y must both be spans: x=(x_lo,x_hi), y=(y_lo,y_hi).")
            if angle is None:
                raise ValueError("For 'sftf' you must provide angle (radians).")
            theta = float(angle)

            # k0, kx, ky (slide "Calculating kx and ky") :contentReference[oaicite:3]{index=3}
            kx = np.cos(theta)
            ky = np.sin(theta)

            # Precompute time-delays δ = (kx x + ky y)/ω for each edge sample (Gaussian / CW plane wave) :contentReference[oaicite:4]{index=4}
            dx, dy = self.dx, self.dy

            # interior indices interpreted as [ix0, ix1), [iy0, iy1)
            # Use cell-center coordinates for delay calculation
            xs = np.arange(ix0, ix1 + 1)
            ys = np.arange(iy0, iy1 + 1)

            Hz_delay_xlo = (kx * ix0 * dx + ky * ys * dy) / self.c0
            Ey_delay_xlo = (kx * (ix0 + 0.5) * dx + ky * ys * dy) / self.c0

            Hz_delay_xhi = (kx * (ix1 + 1) * dx + ky * ys * dy) / self.c0
            Ey_delay_xhi = (kx * (ix1 + 1.5) * dx + ky * ys * dy) / self.c0

            Hz_delay_ylo = (kx * xs * dx + ky * iy0 * dy) / self.c0
            Ex_delay_ylo = (kx * xs * dx + ky * (iy0 + 0.5) * dy) / self.c0

            Hz_delay_yhi = (kx * xs * dx + ky * (iy1 + 1) * dy) / self.c0
            Ex_delay_yhi = (kx * xs * dx + ky * (iy1 + 1.5) * dy) / self.c0

            s["angle"] = theta
            s["Hz_delay_xlo"] = Hz_delay_xlo
            s['Ey_delay_xlo'] = Ey_delay_xlo
            s['Hz_delay_xhi'] = Hz_delay_xhi
            s['Ey_delay_xhi'] = Ey_delay_xhi
            s['Hz_delay_ylo'] = Hz_delay_ylo
            s['Ex_delay_ylo'] = Ex_delay_ylo
            s['Hz_delay_yhi'] = Hz_delay_yhi
            s['Ex_delay_yhi'] = Ex_delay_yhi

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

            Hz_modes, Ex_modes, n_effs = self._wg_modes_y(
                lo, hi, iy_line, f_center,
                num_modes=max(1, int(modes_to_show)),
                guess=eig_guess, amplitude=float(amplitude)
            )
            # Visualize only the Ex mode profiles; title shows n_eff
            # Visualize Ex and Hz profiles; title shows n_eff
            if is_show:
                import matplotlib.pyplot as plt
                lo, hi = (min(ix0, ix1), max(ix0, ix1))
                x_axis = (np.arange(lo, hi) + 0.5) * self.dx  # cell centers

                rows = min(Hz_modes.shape[0], int(modes_to_show))
                fig, axs = plt.subplots(rows, 1, figsize=(8, 2.6 * rows), sharex=True)
                if rows == 1: axs = [axs]

                for m in range(rows):
                    ax1 = axs[m]
                    ax2 = ax1.twinx()
                    ax1.plot(x_axis, Ex_modes[m], linewidth=1.6, label='Ex')
                    ax2.plot(x_axis, Hz_modes[m], linestyle='--', linewidth=1.2, label='Hz')

                    ax1.set_ylabel('Ex (arb.)')
                    ax2.set_ylabel('Hz (arb.)')
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
            mi = min(mi, Hz_modes.shape[0] - 1)
            s['Hz_src'] = Hz_modes[mi]
            s['Ex_src'] = Ex_modes[mi]
            s['n_eff'] = float(n_effs[mi])

        # --- waveguide port (vertical, normal = x) ---
        elif k == 'waveguide-x':
            lo, hi = (min(iy0, iy1), max(iy0, iy1))
            ix_line = ix0
            if f_min is not None and f_max is not None:
                f_center = 0.5 * (f_min + f_max)
            elif f_max is not None:
                f_center = f_max
            else:
                f_center = self.f_max

            Hz_modes, Ey_modes, n_effs = self._wg_modes_x(
                lo, hi, ix_line, f_center,
                num_modes=max(1, int(modes_to_show)),
                guess=eig_guess, amplitude=float(amplitude)
            )

            if is_show:
                import matplotlib.pyplot as plt
                y_axis = (np.arange(lo, hi) + 0.5) * self.dy

                rows = min(Hz_modes.shape[0], int(modes_to_show))
                fig, axs = plt.subplots(rows, 1, figsize=(8, 2.6 * rows), sharex=True)
                if rows == 1:
                    axs = [axs]

                for m in range(rows):
                    ax1 = axs[m]
                    ax2 = ax1.twinx()
                    ax1.plot(y_axis, Ey_modes[m], linewidth=1.6, label='Ey')
                    ax2.plot(y_axis, Hz_modes[m], linestyle='--', linewidth=1.2, label='Hz')

                    ax1.set_ylabel('Ey (arb.)')
                    ax2.set_ylabel('Hz (arb.)')
                    ax1.set_title(f'mode {m + 1}: n_eff = {n_effs[m]:.6f}')
                    ax1.grid(True, alpha=0.25)

                    lines, labels = [], []
                    for a in (ax1, ax2):
                        l, lab = a.get_legend_handles_labels()
                        lines += l
                        labels += lab
                    ax1.legend(lines, labels, loc='upper right')

                axs[-1].set_xlabel('y (m)')
                fig.suptitle(f'waveguide-x port at x={ix_line}')
                fig.tight_layout()
                plt.show()

            mi = max(1, int(mode_index)) - 1
            mi = min(mi, Hz_modes.shape[0] - 1)
            s['Hz_src'] = Hz_modes[mi]
            s['Ey_src'] = Ey_modes[mi]
            s['n_eff'] = float(n_effs[mi])

        self.sources.append(s)

    # --- Line Monitor ---
    def add_line_monitor(self, x, y, t=None):
        """
        Add a 1D monitors aligned to the grid:
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

        self.monitors.append({
            "ix0": ix0, "ix1": ix1,
            "iy0": iy0, "iy1": iy1,
            "it0": it0, "it1": it1,
            "orientation": "horizontal" if is_h else "vertical",
        })

    # ---------- spatial curls ----------
    def calculate_Curl_E(self):
        # identical to your implementations (periodic variants)
        per_x = hasattr(self, "periodic") and ('x' in self.periodic)
        per_y = hasattr(self, "periodic") and ('y' in self.periodic)

        if per_y:
            for nx in range(self.Nx):
                for ny in range(self.Ny - 1):
                    self.d_Ex_y[nx, ny] = (self.Ex[nx, ny + 1] - self.Ex[nx, ny]) / self.dy
                self.d_Ex_y[nx, self.Ny - 1] = (self.Ex[nx, 0] - self.Ex[nx, self.Ny - 1]) / self.dy
        else:
            for nx in range(self.Nx):
                for ny in range(self.Ny - 1):
                    self.d_Ex_y[nx, ny] = (self.Ex[nx, ny + 1] - self.Ex[nx, ny]) / self.dy
                self.d_Ex_y[nx, self.Ny - 1] = (0 - self.Ex[nx, self.Ny - 1]) / self.dy

        if per_x:
            for ny in range(self.Ny):
                for nx in range(self.Nx - 1):
                    self.d_Ey_x[nx, ny] = (self.Ey[nx + 1, ny] - self.Ey[nx, ny]) / self.dx
                self.d_Ey_x[self.Nx - 1, ny] = (self.Ey[0, ny] - self.Ey[self.Nx - 1, ny]) / self.dx
        else:
            for ny in range(self.Ny):
                for nx in range(self.Nx - 1):
                    self.d_Ey_x[nx, ny] = (self.Ey[nx + 1, ny] - self.Ey[nx, ny]) / self.dx
                self.d_Ey_x[self.Nx - 1, ny] = (0 - self.Ey[self.Nx - 1, ny]) / self.dx

    def calcualte_Psi_B(self):
        self.Psi_Bz_x = self.b_Bz_x * self.Psi_Bz_x + self.c_Bz_x * self.d_Ey_x
        self.Psi_Bz_y = self.b_Bz_y * self.Psi_Bz_y + self.c_Bz_y * self.d_Ex_y

    def update_B(self):
        self.Bz = self.Bz - self.M * (
                self.d_Ey_x / self.kappa_x - self.d_Ex_y / self.kappa_y + self.Psi_Bz_x - self.Psi_Bz_y)

    def update_H(self):
        self.Hz = self.Bz / self.MRzz

    def calculate_Curl_H(self):
        per_x = hasattr(self, "periodic") and ('x' in self.periodic)
        per_y = hasattr(self, "periodic") and ('y' in self.periodic)

        if per_y:
            for nx in range(self.Nx):
                for ny in range(1, self.Ny):
                    self.d_Hz_y[nx, ny] = (self.Hz[nx, ny] - self.Hz[nx, ny - 1]) / self.dy
                self.d_Hz_y[nx, 0] = (self.Hz[nx, 0] - self.Hz[nx, self.Ny - 1]) / self.dy
        else:
            for nx in range(self.Nx):
                for ny in range(1, self.Ny):
                    self.d_Hz_y[nx, ny] = (self.Hz[nx, ny] - self.Hz[nx, ny - 1]) / self.dy
                self.d_Hz_y[nx, 0] = (self.Hz[nx, 0] - 0) / self.dy

        if per_x:
            for ny in range(self.Ny):
                for nx in range(1, self.Nx):
                    self.d_Hz_x[nx, ny] = (self.Hz[nx, ny] - self.Hz[nx - 1, ny]) / self.dx
                self.d_Hz_x[0, ny] = (self.Hz[0, ny] - self.Hz[self.Nx - 1, ny]) / self.dx
        else:
            for ny in range(self.Ny):
                for nx in range(1, self.Nx):
                    self.d_Hz_x[nx, ny] = (self.Hz[nx, ny] - self.Hz[nx - 1, ny]) / self.dx
                self.d_Hz_x[0, ny] = (self.Hz[0, ny] - 0) / self.dx

    def calcualte_Psi_D(self):
        self.Psi_Dx_y = self.b_Dx_y * self.Psi_Dx_y + self.c_Dx_y * self.d_Hz_y
        self.Psi_Dy_x = self.b_Dy_x * self.Psi_Dy_x + self.c_Dy_x * self.d_Hz_x

    def update_D(self):
        self.Dx += self.M * (self.d_Hz_y / self.kappa_y + self.Psi_Dx_y)
        self.Dy -= self.M * (self.d_Hz_x / self.kappa_x + self.Psi_Dy_x)

    def update_E(self):
        self.Ex = self.Dx / self.ERxx
        self.Ey = self.Dy / self.ERyy

    # ---------- main loop ----------
    def run(self, record_stride=1, is_include_history=True):
        self._init_Coeff()
        self.is_include_history = is_include_history
        Nx, Ny = self.Nx, self.Ny

        # recording metadata
        self.record_stride = int(record_stride)

        if self.is_include_history:
            Nt_rec = (self.Nt + self.record_stride - 1) // self.record_stride
            self.Nt_rec = int(Nt_rec)
            # allocate histories with compact dtype (float32) to reduce RAM
            dtype_hist = self.Ex.dtype  # or: np.float32
            self.Ex_history = np.zeros((Nt_rec, Nx, Ny), dtype=dtype_hist)
            self.Ey_history = np.zeros((Nt_rec, Nx, Ny), dtype=dtype_hist)
            self.Hz_history = np.zeros((Nt_rec, Nx, Ny), dtype=dtype_hist)
            rec_idx = 0
        else:
            # no history arrays in monitors-only mode
            self.Nt_rec = 0
            rec_idx = None

        # --- monitors: prepare buffers (use existing 'orientation' and unpack with **) ---
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
                    **m,  # inherit all monitors keys
                    "Hz": np.empty((Tm, L), dtype=self.Hz.dtype),
                    "Ex": np.empty((Tm, L), dtype=self.Ex.dtype),
                    "Ey": np.empty((Tm, L), dtype=self.Ey.dtype),
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
                    "Hz": np.empty((Tm, L), dtype=self.Hz.dtype),
                    "Ex": np.empty((Tm, L), dtype=self.Ex.dtype),
                    "Ey": np.empty((Tm, L), dtype=self.Ey.dtype),
                    "_x": ix0,  # fixed x
                    "_sly": slice(iy0, iy1),  # precomputed y-span
                }

            monitor_results.append(buf)

        for t_index in tqdm(range(self.Nt), desc="FDTD simulation", unit="step"):
            # E-curl
            self.calculate_Curl_E()

            # --- SF/TF E injection (by normal) ---
            for s in self.sources:
                if s["kind"] == 'sftf':
                    ix_lo = s["ix0"]
                    ix_hi = s["ix1"]
                    iy_lo = s["iy0"]
                    iy_hi = s["iy1"]

                    nx_side = ix_hi - ix_lo
                    ny_side = iy_hi - iy_lo
                    if nx_side <= 0 or ny_side <= 0:
                        continue

                    t_now = t_index * self.dt
                    kx = np.cos(s["angle"])
                    ky = np.sin(s["angle"])
                    k0 = 1

                    # --- Left edge: Ey tangential, normal = -x ---
                    if ix_lo - 1 >= 0:
                        t_edge = t_now - s["Ey_delay_xlo"]  # vector over j
                        Ey_inc = (kx / k0) * self._g(s, t_edge)  # TEz plane-wave relation
                        for j_off, j in enumerate(range(iy_lo, iy_hi + 1)):
                            self.d_Ey_x[ix_lo - 1, j] -= Ey_inc[j_off] / self.dx

                    # --- Right edge: Ey tangential, normal = +x ---
                    if ix_hi - 1 >= 0 and ix_hi - 1 < self.Nx:
                        t_edge = t_now - s["Ey_delay_xhi"]
                        Ey_inc = (kx / k0) * self._g(s, t_edge)
                        for j_off, j in enumerate(range(iy_lo, iy_hi + 1)):
                            self.d_Ey_x[ix_hi, j] += Ey_inc[j_off] / self.dx

                    # --- Bottom edge: Ex tangential, normal = -y ---
                    if iy_lo - 1 >= 0:
                        t_edge = t_now - s["Ex_delay_ylo"]  # vector over i
                        Ex_inc = -(ky / k0) * self._g(s, t_edge)
                        for i_off, i in enumerate(range(ix_lo, ix_hi + 1)):
                            self.d_Ex_y[i, iy_lo - 1] -= Ex_inc[i_off] / self.dy

                    # --- Top edge: Ex tangential, normal = +y ---
                    if iy_hi - 1 >= 0 and iy_hi - 1 < self.Ny:
                        t_edge = t_now - s["Ex_delay_yhi"]
                        Ex_inc = -(ky / k0) * self._g(s, t_edge)
                        for i_off, i in enumerate(range(ix_lo, ix_hi + 1)):
                            self.d_Ex_y[i, iy_hi] += Ex_inc[i_off] / self.dy

                # E injection (waveguide-y)
                elif s['kind'] == 'waveguide-y':
                    y = s["iy0"]
                    i0, i1 = s["ix0"], s["ix1"]
                    lo, hi = (min(i0, i1), max(i0, i1))
                    E_src = self._g(s, t_index * self.dt)
                    for i in range(lo, hi):
                        if 0 <= y - 1 < self.Ny:
                            self.d_Ex_y[i, y - 1] -= (1.0 / self.dy) * E_src * s["Ex_src"][i - lo]
                elif s['kind'] == 'waveguide-x':
                    x = s["ix0"]
                    j0, j1 = s["iy0"], s["iy1"]
                    lo, hi = (min(j0, j1), max(j0, j1))
                    E_src = self._g(s, t_index * self.dt)
                    for j in range(lo, hi):
                        if 0 <= x - 1 < self.Nx:
                            self.d_Ey_x[x - 1, j] -= (1.0 / self.dx) * E_src * s["Ey_src"][j - lo]

            self.calcualte_Psi_B()
            self.update_B()
            # --- soft sources (point/line-soft) into Dz ---
            t_now = t_index * self.dt
            for s in self.sources:
                if s["kind"] == 'point':
                    i, j = s["ix0"], s["iy0"]
                    self.Bz[i, j] += self._g(s, t_now)  # :contentReference[oaicite:9]{index=9}
                elif s["kind"] == 'line-soft':
                    # If ix span -> horizontal line at y=iy0; else if iy span -> vertical line at x=ix0
                    if s["ix0"] != s["ix1"]:
                        y = s["iy0"]
                        for i in range(min(s["ix0"], s["ix1"]), max(s["ix0"], s["ix1"])):
                            self.Bz[i, y] += self._g(s, t_now)
                    else:
                        x = s["ix0"]
                        for j in range(min(s["iy0"], s["iy1"]), max(s["iy0"], s["iy1"])):
                            self.Bz[x, j] += self._g(s, t_now)

            self.update_H()
            self.calculate_Curl_H()

            # --- SF/TF H injection (TF/SF interface) ---
            for s in self.sources:
                if s["kind"] == 'sftf':
                    ix_lo = s["ix0"]
                    ix_hi = s["ix1"]
                    iy_lo = s["iy0"]
                    iy_hi = s["iy1"]

                    nx_side = ix_hi - ix_lo
                    ny_side = iy_hi - iy_lo
                    if nx_side <= 0 or ny_side <= 0:
                        continue

                    t_half = t_index * self.dt + self.dt / 2.0  # Hz is half-step in time

                    # --- Left edge: uses Hy, normal = -x ---
                    if ix_lo < self.Nx:
                        t_edge = t_half - s["Hz_delay_xlo"]
                        Hz_inc = self._g(s, t_edge)
                        for j_off, j in enumerate(range(iy_lo, iy_hi + 1)):
                            if 0 <= j < self.Ny:
                                self.d_Hz_x[ix_lo, j] -= Hz_inc[j_off] / self.dx

                    # --- Right edge: uses Hy, normal = +x ---
                    if ix_hi < self.Nx:
                        t_edge = t_half - s["Hz_delay_xhi"]
                        Hz_inc = self._g(s, t_edge)
                        for j_off, j in enumerate(range(iy_lo, iy_hi + 1)):
                            if 0 <= j < self.Ny:
                                self.d_Hz_x[ix_hi + 1, j] += Hz_inc[j_off] / self.dx

                    # --- Bottom edge: uses Hx, normal = -y ---
                    if iy_lo < self.Ny:
                        t_edge = t_half - s["Hz_delay_ylo"]
                        Hz_inc = self._g(s, t_edge)
                        for i_off, i in enumerate(range(ix_lo, ix_hi + 1)):
                            if 0 <= i < self.Nx:
                                self.d_Hz_y[i, iy_lo] -= Hz_inc[i_off] / self.dy

                    # --- Top edge: uses Hx, normal = +y ---
                    if iy_hi < self.Ny:
                        t_edge = t_half - s["Hz_delay_yhi"]
                        Hz_inc = self._g(s, t_edge)
                        for i_off, i in enumerate(range(ix_lo, ix_hi + 1)):
                            if 0 <= i < self.Nx:
                                self.d_Hz_y[i, iy_hi + 1] += Hz_inc[i_off] / self.dy

                # H injection (waveguide-y)
                elif s["kind"] == 'waveguide-y':
                    n_eff = s["n_eff"]
                    H_src = -self._g(s, t_index * self.dt + self.dy * n_eff / (2 * self.c0) + self.dt / 2.0)
                    y = s["iy0"]
                    i0, i1 = s["ix0"], s["ix1"]
                    lo, hi = (min(i0, i1), max(i0, i1))
                    for i in range(lo, hi):
                        if 0 <= y < self.Ny:
                            self.d_Hz_y[i, y] += (1.0 / self.dy) * H_src * s["Hz_src"][i - lo]

                elif s["kind"] == 'waveguide-x':
                    n_eff = s["n_eff"]
                    H_src = -self._g(s, t_index * self.dt + self.dx * n_eff / (2 * self.c0) + self.dt / 2.0)
                    x = s["ix0"]
                    j0, j1 = s["iy0"], s["iy1"]
                    lo, hi = (min(j0, j1), max(j0, j1))
                    for j in range(lo, hi):
                        if 0 <= x < self.Nx:
                            self.d_Hz_x[x, j] -= (1.0 / self.dx) * H_src * s["Hz_src"][j - lo]

            self.calcualte_Psi_D()
            self.update_D()
            # update E
            self.update_E()

            # --- capture monitors at this step (no squeeze; direct 1D writes) ---
            for buf in monitor_results:
                if buf["it0"] <= t_index < buf["it1"]:
                    k = t_index - buf["it0"]
                    if buf["orientation"] == "horizontal":
                        self_Hz = self.Hz[buf["_slx"], buf["_y"]]
                        self_Ex = self.Ex[buf["_slx"], buf["_y"]]
                        self_Ey = self.Ey[buf["_slx"], buf["_y"]]
                    else:  # vertical
                        self_Hz = self.Hz[buf["_x"], buf["_sly"]]
                        self_Ex = self.Ex[buf["_x"], buf["_sly"]]
                        self_Ey = self.Ey[buf["_x"], buf["_sly"]]

                    buf["Hz"][k, :] = self_Hz
                    buf["Ex"][k, :] = self_Ex
                    buf["Ey"][k, :] = self_Ey

            # record
            if self.is_include_history and (t_index % self.record_stride) == 0:
                self.Ex_history[rec_idx, :, :] = self.Ex
                self.Ey_history[rec_idx, :, :] = self.Ey
                self.Hz_history[rec_idx, :, :] = self.Hz
                rec_idx += 1

        # --- finalize monitors outputs (drop private helper keys) ---
        self.monitor_results = []
        for buf in monitor_results:
            out = {k: v for k, v in buf.items() if not k.startswith("_")}
            self.monitor_results.append(out)

    def show_animation(self, fps=10, dynamic_clim=True, clim_smooth=0.2, pad=1e-12):
        """
        2x2: [ n-map , Hx ]
             [  Hy   , Ez ]
        Adds red markers/lines for sources and black translucent PML patches.
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        if not hasattr(self, "Ex_history") or self.Ex_history.size == 0:
            raise RuntimeError("No recorded field history. Run sim.run(...) first or load from file.")

        if not hasattr(self, "is_include_history") or self.is_include_history == False:
            raise RuntimeError("No recorded field history. Set sim.is_include_history=True and rerun the simulation.")

        Nx, Ny = self.Nx, self.Ny
        extent = [0, self.x_range, 0, self.y_range]
        eps_avg = 0.5 * (self.ERxx + self.ERyy)
        n_map = np.sqrt(self.MRzz * eps_avg)

        # global clim (fallback)
        if not dynamic_clim:
            vmax_Ex = np.max(np.abs(self.Ex_history)) + pad
            vmax_Ey = np.max(np.abs(self.Ey_history)) + pad
            vmax_Hz = np.max(np.abs(self.Hz_history)) + pad
            vmax_global = max(vmax_Ex, vmax_Ey, vmax_Hz)
            clim_H = (-vmax_global, vmax_global)
            clim_E = (-vmax_global, vmax_global)

        plt.ioff()
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        ax_n, ax_ex = axes[0]
        ax_ey, ax_hz = axes[1]

        # n-map
        im_n = ax_n.imshow(n_map.T, origin="lower", aspect="auto", extent=extent, cmap="viridis")
        im_n.set_clim(np.min(n_map), np.max(n_map))
        fig.colorbar(im_n, ax=ax_n).set_label("n")
        ax_n.set_title("Refractive index")
        ax_n.set_xlabel("x (m)")
        ax_n.set_ylabel("y (m)")

        # Ex / Ey / Hz
        im_ex = ax_ex.imshow(self.Ex_history[0].T, origin="lower", aspect="auto", extent=extent, cmap="jet")
        fig.colorbar(im_ex, ax=ax_ex).set_label("Ex")
        ax_ex.set_title("Ex")
        ax_ex.set_xlabel("x (m)")
        ax_ex.set_ylabel("y (m)")

        im_ey = ax_ey.imshow(self.Ey_history[0].T, origin="lower", aspect="auto", extent=extent, cmap="jet")
        fig.colorbar(im_ey, ax=ax_ey).set_label("Ey")
        ax_ey.set_title("Ey")
        ax_ey.set_xlabel("x (m)")
        ax_ey.set_ylabel("y (m)")

        im_hz = ax_hz.imshow(self.Hz_history[0].T, origin="lower", aspect="auto", extent=extent, cmap="jet")
        fig.colorbar(im_hz, ax=ax_hz).set_label("Hz")
        ax_hz.set_title("Hz")
        ax_hz.set_xlabel("x (m)")
        ax_hz.set_ylabel("y (m)")

        # time text
        time_text = ax_hz.text(0.02, 0.02, "", transform=ax_hz.transAxes,
                               bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

        # --- draw PML patches (black, alpha=0.3) ---
        def add_pml(ax):
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

        add_pml(ax_n)

        # --- draw sources as red markers/lines ---
        def draw_sources(ax):
            for s in self.sources:
                k = s["kind"]

                # TF/SF box: draw all four edges
                if k == "sftf":
                    ix_lo = int(s.get("ix_lo", min(s["ix0"], s["ix1"])))
                    ix_hi = int(s.get("ix_hi", max(s["ix0"], s["ix1"])))
                    iy_lo = int(s.get("iy_lo", min(s["iy0"], s["iy1"])))
                    iy_hi = int(s.get("iy_hi", max(s["iy0"], s["iy1"])))

                    x_lo = ix_lo * self.dx
                    x_hi = ix_hi * self.dx
                    y_lo = iy_lo * self.dy
                    y_hi = iy_hi * self.dy

                    # rectangle edges
                    ax.plot([x_lo, x_hi], [y_lo, y_lo], '-', color='red', lw=2)  # bottom
                    ax.plot([x_lo, x_hi], [y_hi, y_hi], '-', color='red', lw=2)  # top
                    ax.plot([x_lo, x_lo], [y_lo, y_hi], '-', color='red', lw=2)  # left
                    ax.plot([x_hi, x_hi], [y_lo, y_hi], '-', color='red', lw=2)  # right

                else:
                    # convert indices back to meters
                    x0 = s["ix0"] * self.dx
                    x1 = s["ix1"] * self.dx
                    y0 = s["iy0"] * self.dy
                    y1 = s["iy1"] * self.dy

                    if k == "point":
                        ax.plot([x0], [y0], 'o', color='red', ms=5, mew=0)
                    else:
                        # line-soft, waveguides, legacy stuff
                        if s["ix0"] != s["ix1"]:  # horizontal
                            ax.plot([x0, x1], [y0, y0], '-', color='red', lw=2)
                        else:  # vertical
                            ax.plot([x0, x0], [y0, y1], '-', color='red', lw=2)

        draw_sources(ax_n)

        # clims
        if dynamic_clim:
            frame0_max = max(np.max(np.abs(self.Ex_history[0])) + pad,
                             np.max(np.abs(self.Ey_history[0])) + pad,
                             np.max(np.abs(self.Hz_history[0])) + pad)
            smoothed_vmax = frame0_max
            im_ex.set_clim(-smoothed_vmax, smoothed_vmax)
            im_ey.set_clim(-smoothed_vmax, smoothed_vmax)
            im_hz.set_clim(-smoothed_vmax, smoothed_vmax)
        else:
            im_ex.set_clim(*clim_E)
            im_ey.set_clim(*clim_E)
            im_hz.set_clim(*clim_H)

        def _update(frame):
            nonlocal smoothed_vmax
            im_ex.set_data(self.Ex_history[frame].T)
            im_ey.set_data(self.Ey_history[frame].T)
            im_hz.set_data(self.Hz_history[frame].T)
            if dynamic_clim:
                vmax_now = max(np.max(np.abs(self.Ex_history[frame])) + pad,
                               np.max(np.abs(self.Ey_history[frame])) + pad,
                               np.max(np.abs(self.Hz_history[frame])) + pad)
                smoothed_vmax = (1.0 - clim_smooth) * vmax_now + clim_smooth * smoothed_vmax
                v = max(smoothed_vmax, pad)
                im_ex.set_clim(-v, v)
                im_ey.set_clim(-v, v)
                im_hz.set_clim(-v, v)
            tE = frame * getattr(self, "record_stride", 1) * self.dt
            time_text.set_text(f"t = {tE * 1e12:.3f} ps")
            return im_ex, im_ey, im_hz, time_text

        interval_ms = 1000.0 / max(1, fps)
        anim = FuncAnimation(fig, _update, frames=self.Nt_rec, interval=interval_ms, blit=True, repeat=False)
        plt.show()

    def NF2FF(self, top, bottom, left, right, freqs, nphi=361):
        """
        2D NF->FF (TEz / H-mode) using four line monitors (top, bottom, left, right).
        Each monitor index must refer to a line fully in free space (er=mr=1).

        Args
        ----
        top, bottom, left, right : int
            Indices into self.monitor_results for the four sides of the box.
            'top'   : y = y_high (horizontal, x from x1->x2)
            'bottom': y = y_low  (horizontal, x from x1->x2)
            'left'  : x = x_low  (vertical,   y from y1->y2)
            'right' : x = x_high (vertical,   y from y1->y2)
        freqs : 1D array_like
            Frequencies (Hz) at which to compute the FF pattern (θ fixed to 90°, sweep φ).
        nphi : int
            Number of angular samples for φ ∈ [0, 2π).

        Returns
        -------
        ff : dict
            Dictionary with keys:
              - 'phi'   : (nphi,) angles in radians
              - 'freqs' : (Nf,) frequencies
              - 'Etheta','Ephi','Htheta','Hphi' : (Nf, nphi) complex far fields
              - 'Ptheta','Pphi' : (Nf, nphi) time-average power densities
        """
        import numpy as np

        # --- helper: DFT -> phasor vs frequency
        def _phasor_time_series(series, t, freqs):
            # Δt * Σ e^{-j2π f t} f(t)
            t = np.asarray(t, float)  # (T,)
            freqs = np.asarray(freqs, float)  # (Nf,)
            dt = self.dt
            K = np.exp(-1j * 2 * np.pi * freqs[:, None] * t[None, :]) * dt  # (Nf,T)
            return K @ series  # (Nf,L)

        # --- get monitors
        M = self.monitor_results
        mT = M[int(top)]
        mB = M[int(bottom)]
        mL = M[int(left)]
        mR = M[int(right)]

        # Expect orientations consistent with labels
        if not (mT["orientation"] == "horizontal" and
                mB["orientation"] == "horizontal" and
                mL["orientation"] == "vertical" and
                mR["orientation"] == "vertical"):
            raise ValueError("Monitor orientations (top/bottom/left/right) are not as expected.")

        # --- check that each line lies in free space (εr=μr=1)
        def _check_free_space(m):
            if m["orientation"] == "horizontal":
                ix0, ix1 = m["ix0"], m["ix1"]
                y = m["iy0"]
                erx = self.ERxx[ix0:ix1, y]
                ery = self.ERyy[ix0:ix1, y]
                mz = self.MRzz[ix0:ix1, y]
            else:  # vertical
                x = m["ix0"]
                iy0, iy1 = m["iy0"], m["iy1"]
                erx = self.ERxx[x, iy0:iy1]
                ery = self.ERyy[x, iy0:iy1]
                mz = self.MRzz[x, iy0:iy1]

            if not (np.allclose(erx, 1.0) and np.allclose(ery, 1.0) and np.allclose(mz, 1.0)):
                raise ValueError("NF2FF requires monitors entirely in free space (ERxx=ERyy=MRzz=1).")

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
        # In TEz, monitors store Hz, Ex, Ey along the line
        HzT, ExT, EyT = mT["Hz"], mT["Ex"], mT["Ey"]
        HzB, ExB, EyB = mB["Hz"], mB["Ex"], mB["Ey"]
        HzL, ExL, EyL = mL["Hz"], mL["Ex"], mL["Ey"]
        HzR, ExR, EyR = mR["Hz"], mR["Ex"], mR["Ey"]

        # --- phasors at requested freqs: (Nf, L)
        freqs = np.asarray(freqs, float)

        # Out-of-plane magnetic field Hz (scaled by η0, like Ez in TM code)
        HzT_f = _phasor_time_series(HzT, tT, freqs) * self.eta0
        HzB_f = _phasor_time_series(HzB, tB, freqs) * self.eta0
        HzL_f = _phasor_time_series(HzL, tL, freqs) * self.eta0
        HzR_f = _phasor_time_series(HzR, tR, freqs) * self.eta0

        # In-plane electric fields (tangential components on each side)
        ExT_f = _phasor_time_series(ExT, tT, freqs)
        ExB_f = _phasor_time_series(ExB, tB, freqs)
        ExL_f = _phasor_time_series(ExL, tL, freqs)
        ExR_f = _phasor_time_series(ExR, tR, freqs)

        EyT_f = _phasor_time_series(EyT, tT, freqs)
        EyB_f = _phasor_time_series(EyB, tB, freqs)
        EyL_f = _phasor_time_series(EyL, tL, freqs)
        EyR_f = _phasor_time_series(EyR, tR, freqs)

        # --- angular grid
        phi = np.linspace(0.0, 2 * np.pi, int(nphi), endpoint=False)  # (nphi,)
        cφ = np.cos(phi)[None, :]  # (1, nphi)
        sφ = np.sin(phi)[None, :]

        # --- phase factors e^{-jk rhat·r'} for each side (Nf,nφ,L)
        k0 = 2 * np.pi * freqs[:, None] / self.c0  # (Nf,1)

        def _phase_line(xline, yline):
            # rhat·r' = x cosφ + y sinφ
            return np.exp(-1j * (k0[..., None]) * (
                    xline[None, None, :] * cφ[..., None] +
                    yline[None, None, :] * sφ[..., None]
            ))

        PH_T = _phase_line(xT, yT)  # top
        PH_B = _phase_line(xB, yB)  # bottom
        PH_R = _phase_line(xR, yR)  # right
        PH_L = _phase_line(xL, yL)  # left

        # --- integral helpers along x or y
        def _int_x(Fx, PH):  # integrate along x with dx
            return np.sum(Fx[:, None, :] * PH, axis=2) * dx  # (Nf,nφ)

        def _int_y(Fy, PH):  # integrate along y with dy
            return np.sum(Fy[:, None, :] * PH, axis=2) * dy

        # TEz version (dual of TMz):
        # Nθ(φ) from tangential E fields (Ex on horizontal, Ey on vertical)
        Nθ = (- _int_x(ExB_f, PH_B)
              - _int_y(EyR_f, PH_R)
              + _int_x(ExT_f, PH_T)
              + _int_y(EyL_f, PH_L))  # (Nf,nφ)

        # Lφ(φ) from out-of-plane Hz, with trig factors (dual of Ez in TM code)
        Lφ = (- sφ * _int_x(HzB_f, PH_B)
              + cφ * _int_y(HzR_f, PH_R)
              + sφ * _int_x(HzT_f, PH_T)
              - cφ * _int_y(HzL_f, PH_L))  # (Nf,nφ)

        # --- Far fields (θ=90°) for TEz:
        # By duality with TMz expressions:
        #   Hθ = η Nθ + Lφ
        #   Eφ = Lφ/η + Nθ
        eta0 = self.eta0
        Hθ = eta0 * Nθ + Lφ
        Eφ = (Lφ / eta0) + Nθ

        # zeros for the orthogonal components in 2D TEz
        Z = np.zeros_like(Hθ)
        Eθ = Z.copy()
        Hφ = Z.copy()

        # Power densities (per polarization)
        Pθ = (np.abs(Eθ) ** 2) / (2.0 * eta0)  # zero in strict 2D TEz
        Pφ = (np.abs(Eφ) ** 2) / (2.0 * eta0)

        ff = dict(
            phi=phi, freqs=freqs,
            Etheta=Eθ, Ephi=Eφ,
            Htheta=Hθ, Hphi=Hφ,
            Ptheta=Pθ, Pphi=Pφ
        )
        return ff

    def show_FF(self, ff, freq_idx=0, component="Ephi",
                db=True, dr_db=40, normalize="max"):
        """
        Polar plot of a chosen far-field component vs φ.

        Args
        ----
        ff : dict
            Output from NF2FF() with keys: 'phi','freqs', and field/power arrays.
        freq_idx : int or sequence of ints
            Which frequency index/indices to plot (e.g. 0, -1, [0, 10, 20]).
        component : {"Etheta","Ephi","Htheta","Hphi","Ptheta","Pphi"}
            Quantity to plot on the polar axis.
        db : bool
            If True: plot in dB (after normalization).
            If False: plot linear magnitude (optionally normalized).
        dr_db : float
            Dynamic range floor in dB when db=True (e.g., 40 → floor at −40 dB).
        normalize : {"max","integral",None,False}
            How to normalize each curve before plotting.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        phi = np.asarray(ff["phi"])
        freqs = np.asarray(ff["freqs"])
        key = str(component)

        if key not in ff:
            raise KeyError(f"Component '{key}' not found in ff dict.")

        data = np.asarray(ff[key])  # (Nf, nphi)

        # --- parse frequency indices
        if np.isscalar(freq_idx):
            idx_list = [int(freq_idx)]
        else:
            idx_list = [int(i) for i in freq_idx]

        # handle negative indices and clip
        Nf = data.shape[0]
        idx_clean = []
        for i in idx_list:
            if i < 0:
                i = Nf + i
            if 0 <= i < Nf:
                idx_clean.append(i)
        if not idx_clean:
            raise ValueError("No valid freq_idx to plot.")

        # --- normalization helpers
        def _norm_curve(y):
            if not normalize:
                return y
            if normalize == "max":
                m = np.max(np.abs(y))
                return y / m if m > 0 else y
            if normalize == "integral":
                # approximate integral over φ
                integ = np.trapz(np.abs(y), phi)
                return y / integ if integ > 0 else y
            return y

        def _to_db(y):
            mag = np.abs(y)
            # avoid log(0)
            floor_lin = 10 ** (-dr_db / 20.0)
            mag = np.maximum(mag, floor_lin)
            val_db = 20.0 * np.log10(mag)
            return np.maximum(val_db, -dr_db)

        # --- plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="polar")

        handles = []
        labels = []

        for i in idx_clean:
            y = data[i, :]
            y = _norm_curve(y)

            if db:
                r = _to_db(y)
            else:
                r = np.abs(y)
                if normalize:
                    ax.set_rlim(0, 1.0)

            h, = ax.plot(phi, r, lw=1.2)
            handles.append(h)
            labels.append(f"{freqs[i] / 1e9:.3f} GHz")

        # radial limits / labels
        if db:
            ax.set_rlim(-dr_db, 0.0)
            rlabel = "dB (normalized)"
        else:
            rlabel = "Magnitude (normalized)" if normalize else "Magnitude"

        # polar axis formatting
        ax.set_theta_zero_location("E")  # 0° pointing +x
        ax.set_theta_direction(-1)  # increase φ clockwise
        ax.set_thetagrids(np.arange(0, 360, 30))

        # Legend
        ax.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.15, 1.15),
                  framealpha=0.8)

        # Title: what we plotted
        ax.set_title(f"{key} (φ)")

        # radial "axis" label as annotation
        ax.annotate(
            rlabel, xy=(0.98, 0.02), xycoords="axes fraction",
            ha="right", va="bottom", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
        )

        fig.tight_layout()
        plt.show()

    # ---------- state dict / I/O ----------
    def state_dict(self):
        """Return a shallow copy of the simulator's full state dictionary.

        This includes *everything* in self.__dict__ such as materials, fields,
        sources, PML parameters, monitors definitions **and** monitor_results,
        field histories, etc.
        """
        return dict(self.__dict__)

    def load_state_dict(self, state: dict):
        """Replace the simulator's state with a provided dictionary.

        Note: This overwrites the current instance's attributes in-place.
        """
        if not isinstance(state, dict):
            raise TypeError("state must be a dict produced by state_dict() / save().")
        self.__dict__.clear()
        self.__dict__.update(state)

    def save(self, path: str, include_histories: bool = True):
        """Save the full simulator state to *path* using pickle.

        Args
        ----
        path : str
            File path to write (e.g., 'run.pkl').
        include_histories : bool
            If False, large field history arrays (Hx/Hy/Ez/Dz) are stripped to reduce size.
        """
        state = self.state_dict()

        if not include_histories:
            for k in ("Hx_history", "Hy_history", "Ez_history", "Dz_history"):
                if k in state:
                    state[k] = type(state[k])()  # empty like its type

        # Write atomically: write to .part then replace
        import tempfile, os, pickle
        d = os.path.dirname(os.path.abspath(path)) or "."
        base = os.path.basename(path)
        fd, tmp = tempfile.mkstemp(prefix=base + ".part.", dir=d)
        tmp_path = tmp
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
            tmp_path = None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    @classmethod
    def load(cls, path: str):
        """Load a simulator saved with save() and return a ready-to-use instance."""
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        sim = cls.__new__(cls)  # bypass __init__
        if not isinstance(state, dict):
            raise TypeError("Pickle file does not contain a state dict.")
        sim.__dict__.update(state)
        return sim

    @classmethod
    def animate_npz(cls, path, fps=60, dynamic_clim=True, clim_smooth=0.2):
        sim = cls.load(path)
        sim.show_animation(fps=fps, dynamic_clim=dynamic_clim, clim_smooth=clim_smooth)
        return sim
