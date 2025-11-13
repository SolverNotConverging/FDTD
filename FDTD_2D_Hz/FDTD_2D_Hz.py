import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm


class FDTD_2D_Hz:
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
        #   kind: 'point' | 'line-soft' | 'sftf-x' | 'sftf-y'
        #   ix0, ix1, iy0, iy1 (ints; for points, ix0,iy0 used; for lines, spans are used)
        #   amplitude, t0, tw, f_min (or None), f_max
        #   direction: '+x'/'-x'/' +y'/'-y' (used by SF/TF if needed later)
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
            MRzz_obj = float(ER[2])
        else:
            MRzz_obj = float(ER)

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
