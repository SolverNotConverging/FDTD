import numpy as np
from matplotlib.patches import Rectangle
from tqdm import tqdm


class FDTD_2D_Ez:
    """
    2D TMz FDTD with:
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

        # periodic directions string like '' | 'x' | 'y' | 'xy'
        self.periodic = ''

        # materials
        self.ERzz = np.ones((self.Nx, self.Ny))
        self.MRxx = np.ones((self.Nx, self.Ny))
        self.MRyy = np.ones((self.Nx, self.Ny))

        # fields, we use normalized E' D' H' B'

        # H' = sqrt(mu0/eps0)*H, B'=sqrt(eps0*mu0)*B, B'=mu_r*H'
        self.Bx = np.zeros((self.Nx, self.Ny))
        self.Hx = np.zeros((self.Nx, self.Ny))

        self.By = np.zeros((self.Nx, self.Ny))
        self.Hy = np.zeros((self.Nx, self.Ny))

        # E'=E, D'=eps0*D, D'=eps_r*E
        self.Dz = np.zeros((self.Nx, self.Ny))
        self.Ez = np.zeros((self.Nx, self.Ny))

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

        self.b_Bx_y = np.zeros((self.Nx, self.Ny))
        self.b_By_x = np.zeros((self.Nx, self.Ny))
        self.b_Dz_x = np.zeros((self.Nx, self.Ny))
        self.b_Dz_y = np.zeros((self.Nx, self.Ny))

        self.c_By_x = np.zeros((self.Nx, self.Ny))
        self.c_Bx_y = np.zeros((self.Nx, self.Ny))
        self.c_Dz_x = np.zeros((self.Nx, self.Ny))
        self.c_Dz_y = np.zeros((self.Nx, self.Ny))

        # update coefficients
        self.Psi_Bx_y = np.zeros((self.Nx, self.Ny))
        self.Psi_By_x = np.zeros((self.Nx, self.Ny))
        self.Psi_Dz_x = np.zeros((self.Nx, self.Ny))
        self.Psi_Dz_y = np.zeros((self.Nx, self.Ny))

        # curls + integrals
        self.d_Ez_x = np.zeros((self.Nx, self.Ny))
        self.d_Ez_y = np.zeros((self.Nx, self.Ny))
        self.d_Hy_x = np.zeros((self.Nx, self.Ny))
        self.d_Hx_y = np.zeros((self.Nx, self.Ny))

        # multi-source list ---
        # each source is a dict with keys:
        #   kind: 'point' | 'line-soft' | 'sftf' | 'waveguide-x' | 'waveguide-y'
        #   ix0, ix1, iy0, iy1 (ints; for points, ix0,iy0 used; for spans, [ix0,ix1), [iy0,iy1))
        #   amplitude, t0, tw, f_min (or None), f_max
        #   (for 'sftf'):
        #       angle      : propagation angle θ in radians (measured from +x toward +y)
        #       f0         : scalar center frequency used for kx,ky
        #       kx, ky     : components of k-vector in the source region
        #       delay_xlo, delay_xhi : 1D arrays of time delay for the left/right TF/SF edges
        #       delay_ylo, delay_yhi : 1D arrays of time delay for the bottom/top TF/SF edges
        self.sources = []

        self.avg_freqs = []  # one per source (spectral centroid for info/diagnostics)

        # field monitor dictionary
        self.monitors = []
        self.monitor_results = []

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

        # ---------- coefficients ----------

    def _init_Coeff(self):
        # b-coeffs
        ex1 = self.sigma_x / (self.eps0 * self.kappa_x) + self.alpha_x / self.eps0
        self.b_By_x = np.exp(-ex1 * self.dt)

        ex2 = self.sigma_y / (self.eps0 * self.kappa_y) + self.alpha_y / self.eps0
        self.b_Bx_y = np.exp(-ex2 * self.dt)

        # set b=1 exactly outside PML
        mask_x0 = (self.sigma_x == 0) & (self.alpha_x == 0)
        mask_y0 = (self.sigma_y == 0) & (self.alpha_y == 0)
        self.b_By_x[mask_x0] = 1.0
        self.b_Bx_y[mask_y0] = 1.0

        # c-coeffs with safe division; c=0 where sigma=alpha=0
        den1 = self.sigma_x * self.kappa_x + self.alpha_x * self.kappa_x ** 2
        den2 = self.sigma_y * self.kappa_y + self.alpha_y * self.kappa_y ** 2

        self.c_By_x = np.zeros_like(self.sigma_x)
        self.c_Bx_y = np.zeros_like(self.sigma_y)

        good1 = den1 != 0
        good2 = den2 != 0
        self.c_By_x[good1] = (self.sigma_x[good1] / den1[good1]) * (self.b_By_x[good1] - 1.0)
        self.c_Bx_y[good2] = (self.sigma_y[good2] / den2[good2]) * (self.b_Bx_y[good2] - 1.0)

        # D uses the same b/c as B in TMz
        self.b_Dz_x, self.b_Dz_y = self.b_By_x, self.b_Bx_y
        self.c_Dz_x, self.c_Dz_y = self.c_By_x, self.c_Bx_y

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

    def calculate_source_power_fft(self, source_index=0, window='hann', detrend=True):
        """
        Calculate the one-sided FFT source spectrum and an aperture-aware power estimate.

        The previous implementation returned power from only the temporal waveform
        ``g(t)``. Here we additionally scale by a geometry factor so different
        source kinds (point/line/waveguide/TF-SF) report different injected power
        levels.

        Parameters
        ----------
        source_index : int
            Index into ``self.sources``.
        window : str or None
            Optional time window: ``'hann'``/``'hanning'``, ``'hamming'``,
            ``'blackman'``, or ``None``.
        detrend : bool
            If True, remove the mean before FFT.

        Returns
        -------
        dict with keys:
            'source_index'    : selected source index
            'source_kind'     : source kind string
            'freqs'           : one-sided frequencies (Hz)
            'spectrum'        : one-sided complex waveform spectrum ``G(f)``
            'waveform_power'  : waveform-only spectrum ``|G(f)|^2``
            'power'           : geometry-aware source power estimate
            'geometry_factor' : spatial scaling factor used for ``power``
            'waveform'        : time-domain source waveform ``g(t)``
            'time'            : time axis (s)
        """
        if len(self.sources) == 0:
            raise ValueError("No sources available. Add a source before calling calculate_source_power_fft().")
        if not (0 <= int(source_index) < len(self.sources)):
            raise IndexError(f"source_index {source_index} out of range for {len(self.sources)} sources.")

        s = self.sources[int(source_index)]
        t = np.arange(0, self.Nt * self.dt, self.dt)
        g = np.asarray(self._g(s, t), dtype=float)
        Nt = g.shape[0]
        if Nt < 2:
            raise ValueError("Need at least 2 time samples for FFT power calculation.")

        if detrend:
            g = g - np.mean(g)

        if window is None:
            w = np.ones(Nt, dtype=float)
        else:
            key = str(window).lower()
            if key in ('hann', 'hanning'):
                w = np.hanning(Nt)
            elif key == 'hamming':
                w = np.hamming(Nt)
            elif key == 'blackman':
                w = np.blackman(Nt)
            else:
                raise ValueError("window must be one of: None, 'hann'/'hanning', 'hamming', 'blackman'.")

        spectrum = np.fft.rfft(g * w) / Nt
        freqs = np.fft.rfftfreq(Nt, d=self.dt)
        waveform_power = np.abs(spectrum) ** 2

        # Spatial/aperture factor.
        #
        # For soft sources we weight each excited cell by local material
        # properties (requested): eps_r * sqrt(eps_r * mu_r), where mu_r is
        # represented by the geometric mean of in-plane permeability.
        #
        # For waveguide sources we use the summed real modal Poynting product
        # Re(Et * conj(Ht)).
        k = s.get('kind', '')

        def _mu_eff(ix, iy):
            return float(np.sqrt(self.MRxx[ix, iy] * self.MRyy[ix, iy]))

        def _cell_factor(ix, iy):
            eps_r = float(self.ERzz[ix, iy])
            mu_r = _mu_eff(ix, iy)
            return eps_r * np.sqrt(eps_r * mu_r)

        if k == 'point':
            geometry_factor = _cell_factor(int(s['ix0']), int(s['iy0']))
        elif k == 'line-soft':
            if s['ix0'] != s['ix1']:
                y = int(s['iy0'])
                i0, i1 = int(min(s['ix0'], s['ix1'])), int(max(s['ix0'], s['ix1']))
                geometry_factor = float(np.sum([_cell_factor(i, y) for i in range(i0, i1)]))
            else:
                x = int(s['ix0'])
                j0, j1 = int(min(s['iy0'], s['iy1'])), int(max(s['iy0'], s['iy1']))
                geometry_factor = float(np.sum([_cell_factor(x, j) for j in range(j0, j1)]))
        elif k == 'waveguide-y':
            Et = np.asarray(s.get('Ez_src', np.array([1.0])), dtype=complex)
            Ht = np.asarray(s.get('Hx_src', np.array([1.0])), dtype=complex)
            geometry_factor = float(np.sum(np.real(Et * np.conj(Ht))))
        elif k == 'waveguide-x':
            Et = np.asarray(s.get('Ez_src', np.array([1.0])), dtype=complex)
            Ht = np.asarray(s.get('Hy_src', np.array([1.0])), dtype=complex)
            geometry_factor = float(np.sum(np.real(Et * np.conj(Ht))))
        elif k == 'sftf':
            nx = max(int(s['ix1'] - s['ix0']) + 1, 0)
            ny = max(int(s['iy1'] - s['iy0']) + 1, 0)
            geometry_factor = float(2 * nx + 2 * ny)
        else:
            geometry_factor = 1.0

        power = waveform_power * geometry_factor

        return {
            'source_index': int(source_index),
            'source_kind': k,
            'freqs': freqs,
            'spectrum': spectrum,
            'waveform_power': waveform_power,
            'power': power,
            'geometry_factor': geometry_factor,
            'waveform': g,
            'time': t,
        }

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

        ER_vec = np.asarray(self.ERzz[ix, lo:hi], dtype=float)
        MRxx_vec = np.asarray(self.MRxx[ix, lo:hi], dtype=float)
        MRyy_vec = np.asarray(self.MRyy[ix, lo:hi], dtype=float)

        ERzz_diag = spdiags(ER_vec, 0, shape=(Ny, Ny))
        MRxx_inv = spdiags(1.0 / MRxx_vec, 0, shape=(Ny, Ny))
        MRyy_inv = spdiags(1.0 / MRyy_vec, 0, shape=(Ny, Ny))

        d_plus = np.ones(Ny)
        d_minus = -np.ones(Ny)
        DEY = spdiags([d_plus, d_minus], [1, 0], shape=(Ny, Ny)) / (dy * k0)
        DHY = spdiags([d_plus, d_minus], [0, -1], shape=(Ny, Ny)) / (dy * k0)

        A = ERzz_diag + DHY @ (MRyy_inv @ DEY)
        B = MRxx_inv

        if guess is None:
            n_slice = np.sqrt(ER_vec * 0.5 * (MRxx_vec + MRyy_vec))
            n_guess = float(np.max(n_slice))
            guess = max(n_guess ** 2, 1.0)

        k = int(max(1, num_modes))
        evals, evecs = eigs(A, M=B, k=k, sigma=guess)

        n_eff = np.sqrt(np.maximum(evals.real, 0.0))
        order = np.argsort(-n_eff)
        evecs = evecs[:, order]
        n_eff = n_eff[order]

        Ez_modes = []
        Hy_modes = []

        for m in range(evecs.shape[1]):
            Ez = evecs[:, m]

            kmax = np.argmax(np.abs(Ez))
            phase = np.angle(Ez[kmax])
            Ez = (Ez * np.exp(-1j * phase)).real

            Ez = Ez / (np.max(np.abs(Ez)) + 1e-30)

            Hy = -(n_eff[m]) * (MRyy_inv @ Ez)
            Hy = Hy.A.squeeze() if hasattr(Hy, "A") else np.asarray(Hy).squeeze()
            Hy = Hy.real

            Ez_modes.append(Ez)
            Hy_modes.append(Hy)

        Ez_modes = np.asarray(Ez_modes)
        Hy_modes = np.asarray(Hy_modes)

        return np.asarray(Ez_modes), np.asarray(Hy_modes), np.asarray(n_eff, dtype=float)

    # ---------- public API: add_source ----------
    def add_source(self, kind, x, y, amplitude=1.0, t0=None, tw=None, f_min=None, f_max=None,
                   mode_index=0, modes_to_show=4, eig_guess=None, is_show=True, angle=None):
        """
        Add a source.

        kind:
          'point'       : soft point into Dz at (x,y)
          'line-soft'   : soft line into Dz; give x=(ix0,ix1) & y=j or y=(j0,j1) & x=i
          'sftf'        : TF/SF interface rectangle; give x=(x_lo,x_hi) & y=(y_lo,y_hi) and angle
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
            if not np.isclose(self.dx, self.dy):
                raise ValueError("'sftf' source requires square Yee cells (dx == dy).")

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

            Ez_delay_xlo = (kx * ix0 * dx + ky * ys * dy) / self.c0
            Hy_delay_xlo = (kx * (ix0 - 0.5) * dx + ky * ys * dy) / self.c0

            Ez_delay_xhi = (kx * (ix1 + 1) * dx + ky * ys * dy) / self.c0
            Hy_delay_xhi = (kx * (ix1 + 0.5) * dx + ky * ys * dy) / self.c0

            Ez_delay_ylo = (kx * xs * dx + ky * iy0 * dy) / self.c0
            Hx_delay_ylo = (kx * xs * dx + ky * (iy0 - 0.5) * dy) / self.c0

            Ez_delay_yhi = (kx * xs * dx + ky * (iy1 + 1) * dy) / self.c0
            Hx_delay_yhi = (kx * xs * dx + ky * (iy1 + 0.5) * dy) / self.c0

            s["angle"] = theta
            s["Ez_delay_xlo"] = Ez_delay_xlo
            s['Hy_delay_xlo'] = Hy_delay_xlo
            s['Ez_delay_xhi'] = Ez_delay_xhi
            s['Hy_delay_xhi'] = Hy_delay_xhi
            s['Ez_delay_ylo'] = Ez_delay_ylo
            s['Hx_delay_ylo'] = Hx_delay_ylo
            s['Ez_delay_yhi'] = Ez_delay_yhi
            s['Hx_delay_yhi'] = Hx_delay_yhi

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
                    ax1.set_title(f'mode {m}: n_eff = {n_effs[m]:.6f}')
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

            # select which mode to inject (0-based)
            mi = max(0, int(mode_index))
            mi = min(mi, Ez_modes.shape[0] - 1)
            s['Ez_src'] = Ez_modes[mi]
            s['Hx_src'] = Hx_modes[mi]
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

            Ez_modes, Hy_modes, n_effs = self._wg_modes_x(
                lo, hi, ix_line, f_center,
                num_modes=max(1, int(modes_to_show)),
                guess=eig_guess, amplitude=float(amplitude)
            )

            if is_show:
                import matplotlib.pyplot as plt
                y_axis = (np.arange(lo, hi) + 0.5) * self.dy

                rows = min(Ez_modes.shape[0], int(modes_to_show))
                fig, axs = plt.subplots(rows, 1, figsize=(8, 2.6 * rows), sharex=True)
                if rows == 1:
                    axs = [axs]

                for m in range(rows):
                    ax1 = axs[m]
                    ax2 = ax1.twinx()
                    ax1.plot(y_axis, Ez_modes[m], linewidth=1.6, label='Ez')
                    ax2.plot(y_axis, Hy_modes[m], linestyle='--', linewidth=1.2, label='Hy')

                    ax1.set_ylabel('Ez (arb.)')
                    ax2.set_ylabel('Hy (arb.)')
                    ax1.set_title(f'mode {m}: n_eff = {n_effs[m]:.6f}')
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

            mi = max(0, int(mode_index))
            mi = min(mi, Ez_modes.shape[0] - 1)
            s['Ez_src'] = Ez_modes[mi]
            s['Hy_src'] = Hy_modes[mi]
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

        Recorded H-field components are automatically averaged to the Yee cell
        centers and time-aligned with Ez so that all samples represent the same
        physical location and time.
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

    def _avg_with_neighbor(self, arr, axis, periodic, direction):
        """Average a Yee-staggered field with the neighbour shifted by *direction*."""
        if direction not in (-1, 1):
            raise ValueError("direction must be ±1")

        out = np.empty_like(arr)
        if axis == 0:
            if direction == -1:
                out[1:, :] = arr[:-1, :]
                out[0, :] = arr[-1, :] if periodic else 0.0
            else:
                out[:-1, :] = arr[1:, :]
                out[-1, :] = arr[0, :] if periodic else 0.0
        elif axis == 1:
            if direction == -1:
                out[:, 1:] = arr[:, :-1]
                out[:, 0] = arr[:, -1] if periodic else 0.0
            else:
                out[:, :-1] = arr[:, 1:]
                out[:, -1] = arr[:, 0] if periodic else 0.0
        else:
            raise ValueError("axis must be 0 or 1")

        return 0.5 * (arr + out)

    def _avg_with_lower_neighbor(self, arr, axis, periodic):
        """Convenience wrapper for averaging with the neighbour at index -1."""
        return self._avg_with_neighbor(arr, axis, periodic, direction=-1)

    def _avg_with_upper_neighbor(self, arr, axis, periodic):
        """Convenience wrapper for averaging with the neighbour at index +1."""
        return self._avg_with_neighbor(arr, axis, periodic, direction=+1)

    def calculate_line_monitor_power_fft(self, monitor_index, window='hann', detrend=True, normal_sign=1.0):
        """
        Calculate frequency-domain power flow through one line monitor using FFT.

        The monitor must come from ``run(...)`` and therefore contain collocated
        time samples for ``Ez``, ``Hx`` and ``Hy``.

        Args
        ----
        monitor_index : int
            Index into ``self.monitor_results``.
        window : str or None
            Time-domain window applied before FFT. Supported: ``'hann'``,
            ``'hamming'``, ``'blackman'``. Use ``None`` for rectangular window.
        detrend : bool
            If True, remove per-point DC value before FFT.
        normal_sign : float
            Sign of monitor normal direction (+1 or -1).

        Returns
        -------
        dict with keys:
            'freqs'          : positive frequency bins (Hz)
            'power'          : signed real power through the line (W, per FFT bin)
            'complex_power'  : complex line power spectrum
            'power_density'  : complex power density along the line (Nf, Nline)
            'orientation'    : monitor orientation
            'normal_sign'    : copied input
            'monitor_index'  : copied input
        """
        if not self.monitor_results:
            raise RuntimeError("No monitor data found. Run simulation first.")

        m = self.monitor_results[int(monitor_index)]
        ori = m.get("orientation", "").lower()
        if ori not in ("horizontal", "vertical"):
            raise ValueError(f"Unsupported monitor orientation: '{ori}'.")

        Ez = np.asarray(m["Ez"], dtype=float)
        Hx = np.asarray(m["Hx"], dtype=float)
        Hy = np.asarray(m["Hy"], dtype=float)

        if Ez.ndim != 2 or Hx.shape != Ez.shape or Hy.shape != Ez.shape:
            raise ValueError("Monitor arrays must have shape (Nt_monitor, Nline).")

        Nt = Ez.shape[0]
        if Nt < 2:
            raise ValueError("Need at least 2 time samples for FFT power calculation.")

        if window is None:
            w = np.ones(Nt, dtype=float)
        else:
            ws = str(window).lower()
            if ws in ('hann', 'hanning'):
                w = np.hanning(Nt)
            elif ws == 'hamming':
                w = np.hamming(Nt)
            elif ws == 'blackman':
                w = np.blackman(Nt)
            else:
                raise ValueError("window must be one of: None, 'hann', 'hamming', 'blackman'.")

        if detrend:
            Ez = Ez - np.mean(Ez, axis=0, keepdims=True)
            Hx = Hx - np.mean(Hx, axis=0, keepdims=True)
            Hy = Hy - np.mean(Hy, axis=0, keepdims=True)

        Ez_f = np.fft.rfft(Ez * w[:, None], axis=0) / Nt
        Hx_f = np.fft.rfft(Hx * w[:, None], axis=0) / Nt
        Hy_f = np.fft.rfft(Hy * w[:, None], axis=0) / Nt
        freqs = np.fft.rfftfreq(Nt, d=self.dt)

        if ori == 'horizontal':
            # Sy = Ez * Hx / eta0
            power_density = normal_sign * (0.5 / self.eta0) * Ez_f * np.conj(Hx_f)
            dL = self.dx
        else:
            # Sx = -Ez * Hy / eta0
            power_density = normal_sign * (-0.5 / self.eta0) * Ez_f * np.conj(Hy_f)
            dL = self.dy

        complex_power = np.sum(power_density, axis=1) * dL
        power = np.real(complex_power)

        return {
            "freqs": freqs,
            "power": power,
            "complex_power": complex_power,
            "power_density": power_density,
            "orientation": ori,
            "normal_sign": float(normal_sign),
            "monitor_index": int(monitor_index),
        }

    def plot_line_monitor_power_fft(self, power_result, db=False, ref_power=None, f_range=None, ax=None):
        """Plot FFT line-monitor power returned by ``calculate_line_monitor_power_fft``."""
        import matplotlib.pyplot as plt

        f = np.asarray(power_result["freqs"], dtype=float)
        p = np.asarray(power_result["power"], dtype=float)

        mask = np.ones_like(f, dtype=bool)
        if f_range is not None:
            f0, f1 = float(f_range[0]), float(f_range[1])
            if f1 < f0:
                f0, f1 = f1, f0
            mask &= (f >= f0) & (f <= f1)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        else:
            fig = ax.figure

        ff = f[mask]
        pp = p[mask]

        if db:
            eps = 1e-30
            if ref_power is None:
                ref = max(np.max(np.abs(pp)), eps)
            else:
                ref = max(float(ref_power), eps)
            yy = 10.0 * np.log10(np.maximum(np.abs(pp), eps) / ref)
            ylabel = 'Line power (dB)'
        else:
            yy = pp
            ylabel = 'Line power (W, signed)'

        ax.plot(ff / 1e9, yy, lw=1.5)
        if self.f_min is not None:
            ax.set_xlim(self.f_min / 1e9, self.f_max / 1e9)
        else:
            ax.set_xlim(0, self.f_max / 1e9)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.set_title(
            f"FFT power spectrum (monitor {power_result.get('monitor_index', '?')}, {power_result.get('orientation', '?')})")
        fig.tight_layout()
        return fig, ax

    # ---------- spatial curls ----------
    def calculate_Curl_E(self):
        # identical to your implementations (periodic variants)
        per_x = hasattr(self, "periodic") and ('x' in self.periodic)
        per_y = hasattr(self, "periodic") and ('y' in self.periodic)

        if per_y:
            for nx in range(self.Nx):
                for ny in range(self.Ny - 1):
                    self.d_Ez_y[nx, ny] = (self.Ez[nx, ny + 1] - self.Ez[nx, ny]) / self.dy
                self.d_Ez_y[nx, self.Ny - 1] = (self.Ez[nx, 0] - self.Ez[nx, self.Ny - 1]) / self.dy
        else:
            for nx in range(self.Nx):
                for ny in range(self.Ny - 1):
                    self.d_Ez_y[nx, ny] = (self.Ez[nx, ny + 1] - self.Ez[nx, ny]) / self.dy
                self.d_Ez_y[nx, self.Ny - 1] = (0 - self.Ez[nx, self.Ny - 1]) / self.dy

        if per_x:
            for ny in range(self.Ny):
                for nx in range(self.Nx - 1):
                    self.d_Ez_x[nx, ny] = (self.Ez[nx + 1, ny] - self.Ez[nx, ny]) / self.dx
                self.d_Ez_x[self.Nx - 1, ny] = (self.Ez[0, ny] - self.Ez[self.Nx - 1, ny]) / self.dx
        else:
            for ny in range(self.Ny):
                for nx in range(self.Nx - 1):
                    self.d_Ez_x[nx, ny] = (self.Ez[nx + 1, ny] - self.Ez[nx, ny]) / self.dx
                self.d_Ez_x[self.Nx - 1, ny] = (0 - self.Ez[self.Nx - 1, ny]) / self.dx

    def calcualte_Psi_B(self):
        self.Psi_Bx_y = self.b_Bx_y * self.Psi_Bx_y + self.c_Bx_y * self.d_Ez_y
        self.Psi_By_x = self.b_By_x * self.Psi_By_x + self.c_By_x * self.d_Ez_x

    def update_B(self):
        self.Bx -= self.M * (self.d_Ez_y / self.kappa_y + self.Psi_Bx_y)
        self.By += self.M * (self.d_Ez_x / self.kappa_x + self.Psi_By_x)

    def update_H(self):
        self.Hx = self.Bx / self.MRxx
        self.Hy = self.By / self.MRyy

    def calculate_Curl_H(self):
        per_x = hasattr(self, "periodic") and ('x' in self.periodic)
        per_y = hasattr(self, "periodic") and ('y' in self.periodic)

        if per_y:
            for nx in range(self.Nx):
                for ny in range(1, self.Ny):
                    self.d_Hx_y[nx, ny] = (self.Hx[nx, ny] - self.Hx[nx, ny - 1]) / self.dy
                self.d_Hx_y[nx, 0] = (self.Hx[nx, 0] - self.Hx[nx, self.Ny - 1]) / self.dy
        else:
            for nx in range(self.Nx):
                for ny in range(1, self.Ny):
                    self.d_Hx_y[nx, ny] = (self.Hx[nx, ny] - self.Hx[nx, ny - 1]) / self.dy
                self.d_Hx_y[nx, 0] = (self.Hx[nx, 0] - 0) / self.dy

        if per_x:
            for ny in range(self.Ny):
                for nx in range(1, self.Nx):
                    self.d_Hy_x[nx, ny] = (self.Hy[nx, ny] - self.Hy[nx - 1, ny]) / self.dx
                self.d_Hy_x[0, ny] = (self.Hy[0, ny] - self.Hy[self.Nx - 1, ny]) / self.dx
        else:
            for ny in range(self.Ny):
                for nx in range(1, self.Nx):
                    self.d_Hy_x[nx, ny] = (self.Hy[nx, ny] - self.Hy[nx - 1, ny]) / self.dx
                self.d_Hy_x[0, ny] = (self.Hy[0, ny] - 0) / self.dx

    def calcualte_Psi_D(self):
        self.Psi_Dz_x = self.b_Dz_x * self.Psi_Dz_x + self.c_Dz_x * self.d_Hy_x
        self.Psi_Dz_y = self.b_Dz_y * self.Psi_Dz_y + self.c_Dz_y * self.d_Hx_y

    def update_D(self):
        self.Dz = self.Dz + self.M * (
                self.d_Hy_x / self.kappa_x - self.d_Hx_y / self.kappa_y + self.Psi_Dz_x - self.Psi_Dz_y)

    def update_E(self):
        self.Ez = self.Dz / self.ERzz

    # ---------- main loop ----------

    def run(self, record_stride=1, is_include_history=True):
        self._init_Coeff()
        self.is_include_history = is_include_history
        Nx, Ny = self.Nx, self.Ny
        per_x = hasattr(self, "periodic") and ('x' in self.periodic)
        per_y = hasattr(self, "periodic") and ('y' in self.periodic)
        Hx_prev = self.Hx.copy()
        Hy_prev = self.Hy.copy()

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
            self.calculate_Curl_E()

            # --- SF/TF E injection (TF/SF interface) ---
            for s in self.sources:
                if s["kind"] == 'sftf':
                    # TF region interior indices: [ix_lo, ix_hi), [iy_lo, iy_hi)
                    ix_lo = s["ix0"]
                    ix_hi = s["ix1"]
                    iy_lo = s["iy0"]
                    iy_hi = s["iy1"]

                    t_now = t_index * self.dt

                    # --- curl of E: Gaussian TF/SF injection on all four edges

                    # Left edge x = ix_lo  → affects d_Ez_x[ix_lo-1, iy_lo:iy_hi]
                    if ix_lo - 1 >= 0:
                        t_edge = t_now - s["Ez_delay_xlo"]  # shape (ny_side,)
                        Ezsrc_xlo = self._g(s, t_edge)
                        for j_off, j in enumerate(range(iy_lo, iy_hi + 1)):
                            self.d_Ez_x[ix_lo - 1, j] -= Ezsrc_xlo[j_off] / self.dx

                    # Right edge x = ix_hi-1 → use derivative at ix_hi-1
                    if ix_hi - 1 >= 0 and ix_hi - 1 < self.Nx:
                        t_edge = t_now - s["Ez_delay_xhi"]
                        Ezsrc_xhi = self._g(s, t_edge)
                        for j_off, j in enumerate(range(iy_lo, iy_hi + 1)):
                            self.d_Ez_x[ix_hi, j] += Ezsrc_xhi[j_off] / self.dx

                    # Bottom edge y = iy_lo → affects d_Ez_y[ix_lo:ix_hi, iy_lo-1]
                    if iy_lo - 1 >= 0:
                        t_edge = t_now - s["Ez_delay_ylo"]  # shape (nx_side,)
                        Ezsrc_ylo = self._g(s, t_edge)
                        for i_off, i in enumerate(range(ix_lo, ix_hi + 1)):
                            self.d_Ez_y[i, iy_lo - 1] -= Ezsrc_ylo[i_off] / self.dy

                    # Top edge y = iy_hi-1 → use derivative at iy_hi-1
                    if iy_hi - 1 >= 0 and iy_hi - 1 < self.Ny:
                        t_edge = t_now - s["Ez_delay_yhi"]
                        Ezsrc_yhi = self._g(s, t_edge)
                        for i_off, i in enumerate(range(ix_lo, ix_hi + 1)):
                            self.d_Ez_y[i, iy_hi] += Ezsrc_yhi[i_off] / self.dy

                # E injection (waveguide-y)
                elif s['kind'] == 'waveguide-y':
                    y = s["iy0"]
                    i0, i1 = s["ix0"], s["ix1"]
                    lo, hi = (min(i0, i1), max(i0, i1))
                    E_src = self._g(s, t_index * self.dt)
                    for i in range(lo, hi):
                        if 0 <= y - 1 < self.Ny:
                            self.d_Ez_y[i, y - 1] -= (1.0 / self.dy) * E_src * s["Ez_src"][i - lo]
                elif s['kind'] == 'waveguide-x':
                    x = s["ix0"]
                    j0, j1 = s["iy0"], s["iy1"]
                    lo, hi = (min(j0, j1), max(j0, j1))
                    E_src = self._g(s, t_index * self.dt)
                    for j in range(lo, hi):
                        if 0 <= x - 1 < self.Nx:
                            self.d_Ez_x[x - 1, j] -= (1.0 / self.dx) * E_src * s["Ez_src"][j - lo]

            self.calcualte_Psi_B()
            self.update_B()
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

                    t_half = t_index * self.dt + self.dt / 2.0  # H is half-step in time

                    kx = np.cos(s["angle"])
                    ky = np.sin(s["angle"])
                    k0 = 1

                    # From slide: Hx0 = + (ky/k0)*H0,  Hy0 = - (kx/k0)*H0 (Gaussian TF/SF for curl of H) :contentReference[oaicite:11]{index=11}

                    # Left/right edges use Hy, bottom/top use Hx
                    # Left edge
                    if ix_lo < self.Nx:
                        t_edge = t_half - s["Hy_delay_xlo"]
                        Hy_src_xlo = -  (kx / k0) * self._g(s, t_edge)
                        for j_off, j in enumerate(range(iy_lo, iy_hi + 1)):
                            # x-derivative for Hy corresponds to d_Hy_x at ix_lo
                            self.d_Hy_x[ix_lo, j] -= Hy_src_xlo[j_off] / self.dx

                    # Right edge
                    if ix_hi < self.Nx:
                        t_edge = t_half - s["Hy_delay_xhi"]
                        Hy_src_xhi = -  (kx / k0) * self._g(s, t_edge)
                        for j_off, j in enumerate(range(iy_lo, iy_hi + 1)):
                            self.d_Hy_x[ix_hi + 1, j] += Hy_src_xhi[j_off] / self.dx

                    # Bottom edge (uses Hx)
                    if iy_lo < self.Ny:
                        t_edge = t_half - s["Hx_delay_ylo"]
                        Hx_src_ylo = (ky / k0) * self._g(s, t_edge)
                        for i_off, i in enumerate(range(ix_lo, ix_hi + 1)):
                            self.d_Hx_y[i, iy_lo] -= Hx_src_ylo[i_off] / self.dy

                    # Top edge
                    if iy_hi < self.Ny:
                        t_edge = t_half - s["Hx_delay_yhi"]
                        Hx_src_yhi = (ky / k0) * self._g(s, t_edge)
                        for i_off, i in enumerate(range(ix_lo, ix_hi + 1)):
                            self.d_Hx_y[i, iy_hi + 1] += Hx_src_yhi[i_off] / self.dy


                # H injection (waveguide-y)
                elif s["kind"] == 'waveguide-y':
                    n_eff = s["n_eff"]
                    H_src = -self._g(s, t_index * self.dt + self.dy * n_eff / (2 * self.c0) + self.dt / 2.0)
                    y = s["iy0"]
                    i0, i1 = s["ix0"], s["ix1"]
                    lo, hi = (min(i0, i1), max(i0, i1))
                    for i in range(lo, hi):
                        if 0 <= y < self.Ny:
                            self.d_Hx_y[i, y] -= (1.0 / self.dy) * H_src * s["Hx_src"][i - lo]

                elif s["kind"] == 'waveguide-x':
                    n_eff = s["n_eff"]
                    H_src = -self._g(s, t_index * self.dt + self.dx * n_eff / (2 * self.c0) + self.dt / 2.0)
                    x = s["ix0"]
                    j0, j1 = s["iy0"], s["iy1"]
                    lo, hi = (min(j0, j1), max(j0, j1))
                    for j in range(lo, hi):
                        if 0 <= x < self.Nx:
                            self.d_Hy_x[x, j] += (1.0 / self.dx) * H_src * s["Hy_src"][j - lo]

            self.calcualte_Psi_D()
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

            if monitor_results:
                Hx_center = self._avg_with_lower_neighbor(0.5 * (self.Hx + Hx_prev), axis=1, periodic=per_y)
                Hy_center = self._avg_with_lower_neighbor(0.5 * (self.Hy + Hy_prev), axis=0, periodic=per_x)

            # --- capture monitors at this step (no squeeze; direct 1D writes) ---
            for buf in monitor_results:
                if buf["it0"] <= t_index < buf["it1"]:
                    k = t_index - buf["it0"]
                    if buf["orientation"] == "horizontal":
                        buf["Ez"][k, :] = self.Ez[buf["_slx"], buf["_y"]]
                        buf["Hx"][k, :] = Hx_center[buf["_slx"], buf["_y"]]
                        buf["Hy"][k, :] = Hy_center[buf["_slx"], buf["_y"]]
                    else:  # vertical
                        buf["Ez"][k, :] = self.Ez[buf["_x"], buf["_sly"]]
                        buf["Hx"][k, :] = Hx_center[buf["_x"], buf["_sly"]]
                        buf["Hy"][k, :] = Hy_center[buf["_x"], buf["_sly"]]

            if monitor_results:
                Hx_prev[:, :] = self.Hx
                Hy_prev[:, :] = self.Hy

            # record
            if self.is_include_history and (t_index % self.record_stride) == 0:
                self.Hx_history[rec_idx, :, :] = self.Hx
                self.Hy_history[rec_idx, :, :] = self.Hy
                self.Ez_history[rec_idx, :, :] = self.Ez
                rec_idx += 1

        # --- finalize monitors outputs (drop private helper keys) ---
        self.monitor_results = []
        for buf in monitor_results:
            out = {k: v for k, v in buf.items() if not k.startswith("_")}
            self.monitor_results.append(out)

        # ---------- animation ----------

    def show_animation(self, fps=10, dynamic_clim=True, clim_smooth=0.2, pad=1e-12, n_max=5):
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
        im_n.set_clim(np.min(n_map), min(n_max, np.max(n_map)))
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

                # --- TF/SF: draw all four edges of the TF region ---
                if k == "sftf":
                    # use interior TF indices if present, otherwise fall back to ix0/ix1,...
                    ix_lo = int(s.get("ix_lo", min(s["ix0"], s["ix1"])))
                    ix_hi = int(s.get("ix_hi", max(s["ix0"], s["ix1"])))
                    iy_lo = int(s.get("iy_lo", min(s["iy0"], s["iy1"])))
                    iy_hi = int(s.get("iy_hi", max(s["iy0"], s["iy1"])))

                    x_lo = ix_lo * self.dx
                    x_hi = ix_hi * self.dx
                    y_lo = iy_lo * self.dy
                    y_hi = iy_hi * self.dy

                    # draw a rectangle: bottom, top, left, right
                    ax.plot([x_lo, x_hi], [y_lo, y_lo], '-', color='red', lw=2)  # bottom edge
                    ax.plot([x_lo, x_hi], [y_hi, y_hi], '-', color='red', lw=2)  # top edge
                    ax.plot([x_lo, x_lo], [y_lo, y_hi], '-', color='red', lw=2)  # left edge
                    ax.plot([x_hi, x_hi], [y_lo, y_hi], '-', color='red', lw=2)  # right edge

                else:
                    # convert indices back to meters
                    x0 = s["ix0"] * self.dx
                    x1 = s["ix1"] * self.dx
                    y0 = s["iy0"] * self.dy
                    y1 = s["iy1"] * self.dy

                    if k == "point":
                        ax.plot([x0], [y0], 'o', color='red', ms=5, mew=0)
                    else:
                        # line-soft, waveguide, legacy sftf-x/y if still present
                        if s["ix0"] != s["ix1"]:  # horizontal line
                            ax.plot([x0, x1], [y0, y0], '-', color='red', lw=2)
                        else:  # vertical line
                            ax.plot([x0, x0], [y0, y1], '-', color='red', lw=2)

        draw_sources(ax_n)

        # --- draw line monitors as green dashed lines ---
        def draw_monitors(ax):
            if not hasattr(self, "monitors"):
                return
            for m in self.monitors:
                ix0, ix1 = int(m["ix0"]), int(m["ix1"])
                iy0, iy1 = int(m["iy0"]), int(m["iy1"])
                orient = str(m.get("orientation", "")).lower()

                x0 = ix0 * self.dx
                x1 = ix1 * self.dx
                y0 = iy0 * self.dy
                y1 = iy1 * self.dy

                # horizontal monitor: x spans, y fixed
                if orient == "horizontal" or (ix0 != ix1 and iy0 == iy1):
                    ax.plot([x0, x1], [y0, y0], '--', color='green', lw=1.5)
                # vertical monitor: y spans, x fixed
                elif orient == "vertical" or (ix0 == ix1 and iy0 != iy1):
                    ax.plot([x0, x0], [y0, y1], '--', color='green', lw=1.5)

        draw_monitors(ax_n)

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

    def NF2FF(self, top=None, bottom=None, left=None, right=None, freqs=None, nphi=361, src_index=None):
        """
        2D NF->FF (TMz / E-mode) using four line monitors (top, bottom, left, right).
        Each monitor index must refer to a line fully in free space (er=mr=1).
        The monitors are assumed to store Ez, Hx, Hy samples that have already
        been interpolated to the Yee cell centers and time-aligned by run().

        Args
        ----
        top, bottom, left, right : int or None
            Indices into self.monitor_results for the sides of the box.
            'top'  : y = y_high (horizontal, x from x1->x2)
            'bottom': y = y_low  (horizontal, x from x1->x2)
            'left' : x = x_low   (vertical,   y from y1->y2)
            'right': x = x_high  (vertical,   y from y1->y2)
            At least one side must be provided.
        freqs : 1D array_like
            Frequencies (Hz) at which to compute the FF pattern (θ fixed to 90°, sweep φ).
        nphi : int
            Number of φ samples (0..2π). Default 361.
        src_index : int or None
            If not None, index into self.sources that will be used to compute
            the source spectrum G(f) = Fourier{g(t)} for normalizing the far fields.
            If None:
              - if there is exactly one source in self.sources, that one is used;
              - otherwise, no normalization is applied.

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

        if freqs is None:
            raise ValueError("freqs must be provided.")

        side_indices = {"top": top, "bottom": bottom, "left": left, "right": right}
        if not any(v is not None for v in side_indices.values()):
            raise ValueError("At least one of top/bottom/left/right must be provided.")

        # --- get monitors
        M = self.monitor_results
        need_orientation = {"top": "horizontal", "bottom": "horizontal", "left": "vertical", "right": "vertical"}
        side_monitors = {}
        for side, idx in side_indices.items():
            if idx is None:
                side_monitors[side] = None
                continue
            m = M[int(idx)]
            ori = m.get("orientation", "").lower()
            if ori != need_orientation[side]:
                raise ValueError(f"Monitor {idx} must be {need_orientation[side]}, got '{ori}'.")
            side_monitors[side] = m

        # Differential lengths for integration
        dx = self.dx
        dy = self.dy

        freqs = np.asarray(freqs, float)

        # --- prep per-side data
        side_data = {}
        for side, m in side_monitors.items():
            if m is None:
                side_data[side] = None
                continue

            t_side = np.arange(m["it0"], m["it1"]) * self.dt
            e_side = _phasor_time_series(m["Ez"], t_side, freqs) * self.eta0
            hx_side = _phasor_time_series(m["Hx"], t_side, freqs)
            hy_side = _phasor_time_series(m["Hy"], t_side, freqs)

            if side in ("top", "bottom"):
                x_side = np.arange(m["ix0"], m["ix1"], dtype=float) * self.dx
                y_side = np.full_like(x_side, m["iy0"] * self.dy, dtype=float)
                dl = dx
            else:
                y_side = np.arange(m["iy0"], m["iy1"], dtype=float) * self.dy
                x_side = np.full_like(y_side, m["ix0"] * self.dx, dtype=float)
                dl = dy

            side_data[side] = {
                "x": x_side,
                "y": y_side,
                "Ez": e_side,
                "Hx": hx_side,
                "Hy": hy_side,
                "dl": dl,
            }

        # --- φ grid (θ = 90° plane); r-hat = (cosφ, sinφ)
        phi = np.linspace(0.0, 2 * np.pi, int(nphi), endpoint=False)
        cφ = np.cos(phi)[None, :]  # (1, nphi)
        sφ = np.sin(phi)[None, :]

        # --- phase factors e^{-jk rhat·r'} for each side (Nf,nφ,L)
        k0 = 2 * np.pi * freqs[:, None] / self.c0  # (Nf,1)

        def _phase_xy(xline, yline):
            return np.exp(+1j * (k0[..., None]) * (xline[None, None, :] * cφ[..., None] +
                                                   yline[None, None, :] * sφ[..., None]))

        # local adapter to keep math readable
        def _phase_for(side):
            sd = side_data[side]
            if sd is None:
                return None
            xconst = sd["x"]
            yline = sd["y"]
            return _phase_xy(xconst, yline)

        PH_T = _phase_for("top")
        PH_B = _phase_for("bottom")
        PH_R = _phase_for("right")
        PH_L = _phase_for("left")

        # Nθ(φ) =  - ∫ Hx_bottom e^{-jk·r'} dx - ∫ Hy_right e^{-jk·r'} dy
        #           + ∫ Hx_top    e^{-jk·r'} dx + ∫ Hy_left  e^{-jk·r'} dy
        # Lφ(φ) =  - sinφ ∫ Ez_bottom e^{-jk·r'} dx + cosφ ∫ Ez_right e^{-jk·r'} dy
        #           + sinφ ∫ Ez_top    e^{-jk·r'} dx - cosφ ∫ Ez_left  e^{-jk·r'} dy
        #
        # Discretize: sums over samples with dℓ=dx or dy.
        # Shapes:
        #   Hx*_f : (Nf, Lx),  PH_* : (Nf, nφ, Lx)

        def _int_side(side, field_key):
            sd = side_data[side]
            ph = {"top": PH_T, "bottom": PH_B, "left": PH_L, "right": PH_R}[side]
            if sd is None or ph is None:
                return np.zeros((freqs.size, phi.size), dtype=complex)
            return np.sum(sd[field_key][:, None, :] * ph, axis=2) * sd["dl"]

        Nθ = (- _int_side("bottom", "Hx")
              - _int_side("right", "Hy")
              + _int_side("top", "Hx")
              + _int_side("left", "Hy"))  # (Nf,nφ)

        Lφ = (- sφ * _int_side("bottom", "Ez")
              + cφ * _int_side("right", "Ez")
              + sφ * _int_side("top", "Ez")
              - cφ * _int_side("left", "Ez"))  # (Nf,nφ)

        # --- Far fields (θ=90°). From slide:
        # Eθ = j k e^{jk r} / (4π r) ( η Nθ + Lφ )
        # Hφ = j k e^{jk r} / (4π r) ( Lφ/η + Nθ )
        # We omit the common scalar prefactor (j k e^{jkr}/(4πr)) because pattern

        eta0 = self.eta0
        Eθ = k0 * (eta0 * Nθ + Lφ)
        Hφ = k0 * ((Lφ / eta0) + Nθ)

        # --- normalize by source spectrum G(f) = Fourier{g(t)} if available/requested ---
        G = None  # store spectrum if we compute it

        src_to_use = None
        if hasattr(self, "sources") and len(self.sources) > 0:
            if src_index is None:
                # auto-pick only if exactly one source
                if len(self.sources) == 1:
                    src_to_use = self.sources[0]
            else:
                src_index = int(src_index)
                if not (0 <= src_index < len(self.sources)):
                    raise IndexError(f"src_index {src_index} out of range for {len(self.sources)} sources.")
                src_to_use = self.sources[src_index]

        if src_to_use is not None:
            t_src = np.arange(self.Nt) * self.dt  # (Nt,)
            g_t = self._g(src_to_use, t_src)  # (Nt,)

            # Same DFT convention as for fields
            G = _phasor_time_series(g_t[:, None], t_src, freqs).ravel()  # (Nf,)

            magG = np.abs(G)
            if np.any(magG):
                Gmax = magG.max()
                if Gmax > 0.0:
                    thresh = 1e-6 * Gmax
                    mask_good = magG >= thresh  # (Nf,)

                    # divide only where G(f) is "good"
                    if np.any(mask_good):
                        Eθ[mask_good, :] /= G[mask_good, None]
                        Hφ[mask_good, :] /= G[mask_good, None]

                    # for frequencies where source has negligible energy,
                    # just zero out the fields (no meaningful normalization)
                    Eθ[~mask_good, :] = 0.0
                    Hφ[~mask_good, :] = 0.0

        # zeros for the orthogonal components in 2D TMz
        Z = np.zeros_like(Eθ)
        Eφ = Z.copy()
        Hθ = Z.copy()

        # Power densities (per polarization)
        # NOTE: Pθ is now effectively divided by |G(f)|^2 because Eθ was divided by G(f).
        Pθ = 1 / 2 * (Eθ * Hφ)
        Pφ = Z.copy()  # zero

        ff = dict(
            phi=phi, freqs=freqs,
            Etheta=Eθ, Ephi=Eφ,
            Htheta=Hθ, Hphi=Hφ,
            Ptheta=Pθ, Pphi=Pφ,
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
            Normalization before plotting:
              - "max"      : divide by global max(|y|) over φ across all selected beams
              - "integral" : divide by sqrt(mean(|y|^2)) over φ (RMS) per curve
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

        # clip/validate indices
        Nf = data.shape[0]
        for i in idx_list:
            if not (-Nf <= i < Nf):
                raise IndexError(f"freq_idx {i} out of range for Nf={Nf}")

        # Effective (non-negative) indices
        idx_eff = [i % Nf for i in idx_list]

        # Decide dB rule: fields use 20*log10, powers use 10*log10
        is_power = key.startswith("P")
        log_factor = 10.0 if is_power else 20.0

        # --- global max over all selected beams for normalize == "max"
        if normalize == "max":
            sel = data[idx_eff, :]  # (N_sel, nphi)
            # ignore NaNs when computing max
            with np.errstate(invalid="ignore"):
                global_max = np.nanmax(np.abs(sel))
            if (not np.isfinite(global_max)) or global_max <= 0:
                global_max = None
        else:
            global_max = None

        # Normalization helper
        def _norm_curve(y):
            y = np.asarray(y)
            if normalize == "max":
                if global_max is None:
                    return y
                return y / global_max
            elif normalize == "integral":
                # RMS per curve, again ignoring NaNs
                with np.errstate(invalid="ignore"):
                    s = np.sqrt(np.nanmean(np.abs(y) ** 2))
                if (not np.isfinite(s)) or s == 0:
                    s = 1.0
                return y / s
            else:
                return y

        # dB conversion with floor (NO per-curve renormalization here)
        def _to_db(y):
            y = np.abs(y)
            floor_lin = 10 ** (-dr_db / log_factor)
            y = np.maximum(y, floor_lin)
            val_db = log_factor * np.log10(y)
            return np.maximum(val_db, -dr_db)

        # --- figure: single polar plot
        fig = plt.figure(figsize=(6.8, 5.4))
        ax = plt.subplot(1, 1, 1, projection='polar')
        ax.grid(True, alpha=0.3)
        ax.set_theta_zero_location("E")  # 0° at +x (East)
        ax.set_theta_direction(1)  # CCW

        handles = []
        labels = []

        # plot each requested frequency slice
        for i, j in zip(idx_list, idx_eff):
            y = data[j, :]  # (nphi,)
            y = _norm_curve(y)  # apply chosen (global) normalization

            if db:
                r = _to_db(y)
                ax.set_rlim(-dr_db, 0)  # fixed dB range
                rlabel = "dB"
                if normalize:
                    rlabel += " (normalized)"
            else:
                r = np.abs(y)
                if normalize:
                    ax.set_rlim(0, 1.0)
                    rlabel = "Magnitude (normalized)"
                else:
                    rlabel = "Magnitude"

            h, = ax.plot(phi, r, lw=1, ls='-')
            handles.append(h)
            labels.append(f"{freqs[j] / 1e9:.3f} GHz")

        ax.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.2, 1.10))
        ax.set_rlabel_position(135)

        # Title: what we plotted + "(φ)"
        ax.set_title(f"{key} (φ)")

        # radial label via annotation
        ax.annotate(rlabel, xy=(0.98, 0.02), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

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
