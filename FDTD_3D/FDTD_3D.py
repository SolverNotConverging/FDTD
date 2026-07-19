"""Three-dimensional finite-difference time-domain solver.

The solver uses the conventional Cartesian Yee lattice.  Material geometry is
rasterized on ``(Nx, Ny, Nz)`` voxels and averaged to the staggered field
locations.  The optional Cython extension advances the complete time loop;
the NumPy implementation is deliberately kept equivalent as a portable
fallback and reference implementation.
"""

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np

try:
    from . import _cython_kernel_3d as _cython_kernel
except (ImportError, ValueError):
    try:
        import _cython_kernel_3d as _cython_kernel
    except ImportError:
        _cython_kernel = None


def _triple(value, name, positive=False, nonnegative=False):
    values = np.asarray(value if np.ndim(value) else (value, value, value), dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite scalar or length-three sequence.")
    if positive and np.any(values <= 0.0):
        raise ValueError(f"{name} must be positive.")
    if nonnegative and np.any(values < 0.0):
        raise ValueError(f"{name} must be non-negative.")
    return tuple(float(v) for v in values)


@dataclass(frozen=True)
class Material:
    """A diagonal, possibly lossy electromagnetic material.

    ``epsilon_r``, ``mu_r``, ``sigma_e`` and ``sigma_m`` may each be a scalar
    or an ``(x, y, z)`` sequence.  Perfect conductors use ``kind='PEC'`` or
    ``kind='PMC'`` and are constrained directly rather than represented by an
    artificial permittivity/permeability.
    """

    name: str
    epsilon_r: tuple = (1.0, 1.0, 1.0)
    mu_r: tuple = (1.0, 1.0, 1.0)
    sigma_e: tuple = (0.0, 0.0, 0.0)
    sigma_m: tuple = (0.0, 0.0, 0.0)
    kind: str = "ordinary"

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Material name must be a non-empty string.")
        kind = str(self.kind).upper()
        if kind not in {"ORDINARY", "PEC", "PMC"}:
            raise ValueError("kind must be 'ordinary', 'PEC', or 'PMC'.")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "epsilon_r", _triple(self.epsilon_r, "epsilon_r", positive=True))
        object.__setattr__(self, "mu_r", _triple(self.mu_r, "mu_r", positive=True))
        object.__setattr__(self, "sigma_e", _triple(self.sigma_e, "sigma_e", nonnegative=True))
        object.__setattr__(self, "sigma_m", _triple(self.sigma_m, "sigma_m", nonnegative=True))
        object.__setattr__(self, "kind", "ordinary" if kind == "ORDINARY" else kind)


class FDTD_3D:
    """3D Yee-grid FDTD with CFS-CPML, geometry, monitors and NF2FF."""

    eps0 = 8.8541878128e-12
    mu0 = 4.0e-7 * np.pi
    c0 = 1.0 / np.sqrt(eps0 * mu0)
    eta0 = np.sqrt(mu0 / eps0)

    @staticmethod
    def suggest_dx_dt(max_epsilon_r, max_mu_r, f_max, cells_per_wavelength=20,
                      courant_factor=0.95, time_samples_per_period=30):
        """Return accuracy-oriented cubic cell size and stable time step."""
        if min(max_epsilon_r, max_mu_r, f_max, cells_per_wavelength,
               courant_factor, time_samples_per_period) <= 0:
            raise ValueError("All arguments must be positive.")
        nmax = np.sqrt(max_epsilon_r * max_mu_r)
        wavelength = FDTD_3D.c0 / (f_max * nmax)
        spacing = wavelength / cells_per_wavelength
        dt_cfl = spacing / (FDTD_3D.c0 * np.sqrt(3.0))
        return {
            "dx": spacing, "dy": spacing, "dz": spacing,
            "dt": min(courant_factor * dt_cfl,
                      1.0 / (time_samples_per_period * f_max)),
            "lambda_min": wavelength, "refractive_index_max": nmax,
        }

    def __init__(self, x_range, y_range, z_range, Nx, Ny, Nz, f_max, Nt,
                 f_min=None, dt=None, subpixel=4, dtype=np.float64):
        self.x_range, self.y_range, self.z_range = map(
            float, (x_range, y_range, z_range))
        self.Nx, self.Ny, self.Nz = map(int, (Nx, Ny, Nz))
        if min(self.x_range, self.y_range, self.z_range) <= 0.0:
            raise ValueError("Domain ranges must be positive.")
        if min(self.Nx, self.Ny, self.Nz) < 2:
            raise ValueError("Nx, Ny and Nz must each be at least two.")
        self.dx = self.x_range / self.Nx
        self.dy = self.y_range / self.Ny
        self.dz = self.z_range / self.Nz
        self.Nt = int(Nt)
        self.f_max = float(f_max)
        self.f_min = None if f_min is None else float(f_min)
        if self.Nt < 1 or self.f_max <= 0.0:
            raise ValueError("Nt and f_max must be positive.")
        if not isinstance(subpixel, (int, np.integer)) or subpixel < 1:
            raise ValueError("subpixel must be a positive integer.")
        self.subpixel = int(subpixel)
        self.dtype = np.dtype(dtype)
        if self.dtype != np.float64:
            raise ValueError("The current Python and Cython kernels require float64.")

        dt_cfl = 1.0 / (self.c0 * np.sqrt(
            self.dx ** -2 + self.dy ** -2 + self.dz ** -2))
        dt_sample = 1.0 / (20.0 * self.f_max)
        self.dt = float(dt) if dt is not None else min(0.99 * dt_cfl, dt_sample)
        if not (0.0 < self.dt <= dt_cfl * (1.0 + 1e-12)):
            raise ValueError(f"dt must be positive and no greater than the CFL limit {dt_cfl:.6g} s.")
        self.dt_cfl = dt_cfl

        shape = (self.Nx, self.Ny, self.Nz)
        self.materials = {}
        self.add_material("vacuum")
        self.add_material("PEC", kind="PEC")
        self.add_material("PMC", kind="PMC")
        self._er = [np.ones(shape) for _ in range(3)]
        self._mr = [np.ones(shape) for _ in range(3)]
        self._sigma_e = [np.zeros(shape) for _ in range(3)]
        self._sigma_m = [np.zeros(shape) for _ in range(3)]
        self.PEC_cells = np.zeros(shape, dtype=bool)
        self.PMC_cells = np.zeros(shape, dtype=bool)

        # E lives on edges; H lives on faces of each voxel.
        self.Ex = np.zeros((self.Nx, self.Ny + 1, self.Nz + 1))
        self.Ey = np.zeros((self.Nx + 1, self.Ny, self.Nz + 1))
        self.Ez = np.zeros((self.Nx + 1, self.Ny + 1, self.Nz))
        self.Hx = np.zeros((self.Nx + 1, self.Ny, self.Nz))
        self.Hy = np.zeros((self.Nx, self.Ny + 1, self.Nz))
        self.Hz = np.zeros((self.Nx, self.Ny, self.Nz + 1))

        self.sources = []
        self.monitors = []
        self.monitor_results = []
        self.current_step = 0
        self.pml_width = (0, 0, 0)
        self._set_identity_cpml()
        self._refresh_material_state()
        self.config("cpu")

    # ------------------------------------------------------------------ materials
    def config(self, backend="cpu"):
        """Select ``cpu`` (compiled when available) or ``python``."""
        key = str(backend).lower()
        if key not in {"cpu", "python", "cython"}:
            raise ValueError("backend must be 'cpu', 'cython', or 'python'.")
        if key == "cython" and _cython_kernel is None:
            raise RuntimeError("The 3D Cython extension is not built. Run setup_cython.py build_ext --inplace.")
        self.backend_requested = key
        self.backend = "cython" if key in {"cpu", "cython"} and _cython_kernel is not None else "python"
        if key == "cpu" and _cython_kernel is None:
            warnings.warn("3D Cython extension unavailable; using the NumPy fallback.", RuntimeWarning)
        return self

    def add_material(self, name, epsilon_r=1.0, mu_r=1.0, sigma_e=0.0,
                     sigma_m=0.0, kind="ordinary"):
        """Define and return a named material for later geometry calls."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Material name must be a non-empty string.")
        kind = str(kind).upper()
        if kind not in {"ORDINARY", "PEC", "PMC"}:
            raise ValueError("kind must be 'ordinary', 'PEC', or 'PMC'.")
        er = _triple(epsilon_r, "epsilon_r", positive=True)
        mr = _triple(mu_r, "mu_r", positive=True)
        se = _triple(sigma_e, "sigma_e", nonnegative=True)
        sm = _triple(sigma_m, "sigma_m", nonnegative=True)
        material = Material(name.strip(), er, mr, se, sm,
                            "ordinary" if kind == "ORDINARY" else kind)
        self.materials[material.name.lower()] = material
        return material

    def get_material(self, material):
        if isinstance(material, Material):
            return material
        try:
            return self.materials[str(material).lower()]
        except KeyError as exc:
            raise KeyError(f"Unknown material {material!r}; define it with add_material first.") from exc

    @staticmethod
    def _edge_average(values, component):
        axes = [axis for axis in range(3) if axis != component]
        padded = np.pad(values, [(0, 0) if a == component else (1, 1)
                                 for a in range(3)], mode="edge")
        result = 0.0
        for first in (0, 1):
            for second in (0, 1):
                slices = [slice(None)] * 3
                for axis, offset in zip(axes, (first, second)):
                    slices[axis] = slice(offset, offset + values.shape[axis] + 1)
                result = result + padded[tuple(slices)]
        return np.ascontiguousarray(result * 0.25)

    @staticmethod
    def _face_average(values, component):
        pads = [(0, 0)] * 3
        pads[component] = (1, 1)
        padded = np.pad(values, pads, mode="edge")
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[component] = slice(0, values.shape[component] + 1)
        hi[component] = slice(1, values.shape[component] + 2)
        return np.ascontiguousarray(0.5 * (padded[tuple(lo)] + padded[tuple(hi)]))

    @staticmethod
    def _edge_mask(values, component):
        axes = [axis for axis in range(3) if axis != component]
        padded = np.pad(values, [(0, 0) if a == component else (1, 1)
                                 for a in range(3)], mode="constant")
        result = None
        for first in (0, 1):
            for second in (0, 1):
                slices = [slice(None)] * 3
                for axis, offset in zip(axes, (first, second)):
                    slices[axis] = slice(offset, offset + values.shape[axis] + 1)
                item = padded[tuple(slices)]
                result = item.copy() if result is None else result | item
        return result

    @staticmethod
    def _face_mask(values, component):
        pads = [(0, 0)] * 3
        pads[component] = (1, 1)
        padded = np.pad(values, pads, mode="constant")
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[component] = slice(0, values.shape[component] + 1)
        hi[component] = slice(1, values.shape[component] + 2)
        return padded[tuple(lo)] | padded[tuple(hi)]

    def _refresh_material_state(self):
        self.ERxx, self.ERyy, self.ERzz = self._er
        self.MRxx, self.MRyy, self.MRzz = self._mr
        self.epsilon_Ex = self.eps0 * self._edge_average(self._er[0], 0)
        self.epsilon_Ey = self.eps0 * self._edge_average(self._er[1], 1)
        self.epsilon_Ez = self.eps0 * self._edge_average(self._er[2], 2)
        self.mu_Hx = self.mu0 * self._face_average(self._mr[0], 0)
        self.mu_Hy = self.mu0 * self._face_average(self._mr[1], 1)
        self.mu_Hz = self.mu0 * self._face_average(self._mr[2], 2)
        sigma_e = [self._edge_average(self._sigma_e[c], c) for c in range(3)]
        sigma_m = [self._face_average(self._sigma_m[c], c) for c in range(3)]
        eps = [self.epsilon_Ex, self.epsilon_Ey, self.epsilon_Ez]
        mu = [self.mu_Hx, self.mu_Hy, self.mu_Hz]
        e_masks = [self._edge_mask(self.PEC_cells, c) for c in range(3)]
        h_masks = [self._face_mask(self.PMC_cells, c) for c in range(3)]
        # Tangential electric components on the terminating outer wall are zero.
        e_masks[0][:, 0, :] = True; e_masks[0][:, -1, :] = True
        e_masks[0][:, :, 0] = True; e_masks[0][:, :, -1] = True
        e_masks[1][0, :, :] = True; e_masks[1][-1, :, :] = True
        e_masks[1][:, :, 0] = True; e_masks[1][:, :, -1] = True
        e_masks[2][0, :, :] = True; e_masks[2][-1, :, :] = True
        e_masks[2][:, 0, :] = True; e_masks[2][:, -1, :] = True
        self.PEC_Ex, self.PEC_Ey, self.PEC_Ez = e_masks
        self.PMC_Hx, self.PMC_Hy, self.PMC_Hz = h_masks
        self.CaEx, self.CbEx = self._loss_coeff(eps[0], sigma_e[0], e_masks[0])
        self.CaEy, self.CbEy = self._loss_coeff(eps[1], sigma_e[1], e_masks[1])
        self.CaEz, self.CbEz = self._loss_coeff(eps[2], sigma_e[2], e_masks[2])
        self.CaHx, self.CbHx = self._loss_coeff(mu[0], sigma_m[0], h_masks[0])
        self.CaHy, self.CbHy = self._loss_coeff(mu[1], sigma_m[1], h_masks[1])
        self.CaHz, self.CbHz = self._loss_coeff(mu[2], sigma_m[2], h_masks[2])

    def _loss_coeff(self, constitutive, conductivity, mask):
        ratio = conductivity * self.dt / (2.0 * constitutive)
        ca = np.ascontiguousarray((1.0 - ratio) / (1.0 + ratio))
        cb = np.ascontiguousarray((self.dt / constitutive) / (1.0 + ratio))
        ca[mask] = 0.0
        cb[mask] = 0.0
        return ca, cb

    def _coord(self, value, axis):
        step = (self.dx, self.dy, self.dz)[axis]
        return float(value * step) if isinstance(value, (int, np.integer)) else float(value)

    def _bounds(self, span, axis):
        if not isinstance(span, (tuple, list, np.ndarray)) or len(span) != 2:
            raise TypeError("Each geometry extent must be a two-element span.")
        a, b = self._coord(span[0], axis), self._coord(span[1], axis)
        a, b = min(a, b), max(a, b)
        limit = (self.x_range, self.y_range, self.z_range)[axis]
        return max(0.0, a), min(limit, b)

    def _paint(self, material, predicate, bounds, subpixel=None):
        mat = self.get_material(material)
        ranges = [range(max(0, int(np.floor(bounds[a][0] / (self.dx, self.dy, self.dz)[a]))),
                        min((self.Nx, self.Ny, self.Nz)[a],
                            int(np.ceil(bounds[a][1] / (self.dx, self.dy, self.dz)[a]))))
                  for a in range(3)]
        if any(len(r) == 0 for r in ranges):
            return self
        if mat.kind in {"PEC", "PMC"}:
            x = (np.asarray(list(ranges[0])) + 0.5) * self.dx
            y = (np.asarray(list(ranges[1])) + 0.5) * self.dy
            z = (np.asarray(list(ranges[2])) + 0.5) * self.dz
            local = predicate(x[:, None, None], y[None, :, None], z[None, None, :])
            cells = np.zeros_like(self.PEC_cells)
            cells[np.ix_(list(ranges[0]), list(ranges[1]), list(ranges[2]))] = local
            if mat.kind == "PEC":
                self.PEC_cells[cells] = True; self.PMC_cells[cells] = False
            else:
                self.PMC_cells[cells] = True; self.PEC_cells[cells] = False
        else:
            ns = self.subpixel if subpixel is None else int(subpixel)
            if ns < 1:
                raise ValueError("subpixel must be positive.")
            offsets = (np.arange(ns) + 0.5) / ns
            ix, iy, iz = tuple(np.asarray(list(items), dtype=int) for items in ranges)
            ys = ((iy[:, None] + offsets[None, :]).reshape(-1) * self.dy)
            zs = ((iz[:, None] + offsets[None, :]).reshape(-1) * self.dz)
            fractions = np.empty((len(ix), len(iy), len(iz)), dtype=float)
            # Vectorize each complete yz slab. This avoids a Python loop per
            # voxel without constructing a potentially enormous 3D subgrid.
            for local_i, i in enumerate(ix):
                xs = (i + offsets) * self.dx
                sampled = np.asarray(predicate(xs[:, None, None], ys[None, :, None],
                                               zs[None, None, :]), dtype=float)
                fractions[local_i] = sampled.reshape(
                    ns, len(iy), ns, len(iz), ns).mean(axis=(0, 2, 4))
            region = np.ix_(ix, iy, iz)
            for c in range(3):
                self._er[c][region] = ((1-fractions)*self._er[c][region] + fractions*mat.epsilon_r[c])
                self._mr[c][region] = ((1-fractions)*self._mr[c][region] + fractions*mat.mu_r[c])
                self._sigma_e[c][region] = ((1-fractions)*self._sigma_e[c][region] + fractions*mat.sigma_e[c])
                self._sigma_m[c][region] = ((1-fractions)*self._sigma_m[c][region] + fractions*mat.sigma_m[c])
            occupied = fractions > 0.0
            pec = self.PEC_cells[region]; pec[occupied] = False; self.PEC_cells[region] = pec
            pmc = self.PMC_cells[region]; pmc[occupied] = False; self.PMC_cells[region] = pmc
        self._refresh_material_state()
        return self

    def add_block(self, material, x, y, z, subpixel=None):
        """Add an axis-aligned block. Spans accept cell indices or metres."""
        bounds = (self._bounds(x, 0), self._bounds(y, 1), self._bounds(z, 2))
        return self._paint(material, lambda X, Y, Z:
                           (X >= bounds[0][0]) & (X < bounds[0][1]) &
                           (Y >= bounds[1][0]) & (Y < bounds[1][1]) &
                           (Z >= bounds[2][0]) & (Z < bounds[2][1]), bounds, subpixel)

    def add_sphere(self, material, center, radius, subpixel=None):
        """Add a sphere whose center/radius are in metres (integer centers are indices)."""
        if len(center) != 3 or radius <= 0:
            raise ValueError("center must have length three and radius must be positive.")
        c = tuple(self._coord(center[a], a) for a in range(3))
        r = float(radius)
        limits = (self.x_range, self.y_range, self.z_range)
        bounds = tuple((max(0, c[a]-r), min(limits[a], c[a]+r)) for a in range(3))
        return self._paint(material, lambda X, Y, Z:
                           (X-c[0])**2 + (Y-c[1])**2 + (Z-c[2])**2 <= r*r,
                           bounds, subpixel)

    def add_cylinder(self, material, center, radius, height=None, axis="z", subpixel=None):
        """Add a finite cylinder centered at ``center`` and aligned with ``axis``."""
        axis = str(axis).lower()
        if axis not in "xyz" or len(center) != 3 or radius <= 0:
            raise ValueError("axis must be x/y/z, center length three, and radius positive.")
        a = "xyz".index(axis)
        c = tuple(self._coord(center[q], q) for q in range(3))
        limits = (self.x_range, self.y_range, self.z_range)
        h = limits[a] if height is None else float(height)
        if h <= 0:
            raise ValueError("height must be positive.")
        radial = [q for q in range(3) if q != a]
        bounds = []
        for q in range(3):
            half = h/2 if q == a else float(radius)
            bounds.append((max(0, c[q]-half), min(limits[q], c[q]+half)))
        def inside(X, Y, Z):
            Q = (X, Y, Z)
            return ((Q[radial[0]]-c[radial[0]])**2 +
                    (Q[radial[1]]-c[radial[1]])**2 <= radius**2) & (np.abs(Q[a]-c[a]) <= h/2)
        return self._paint(material, inside, tuple(bounds), subpixel)

    # ---------------------------------------------------------------------- CPML
    def _set_identity_cpml(self):
        self._pml = {}
        for name, count in zip("xyz", (self.Nx, self.Ny, self.Nz)):
            self._pml[name] = {
                "node_k": np.ones(count + 1), "node_b": np.ones(count + 1), "node_c": np.zeros(count + 1),
                "cell_k": np.ones(count), "cell_b": np.ones(count), "cell_c": np.zeros(count),
            }
        self._allocate_psi()

    def _allocate_psi(self):
        for name, field in (("Psi_Hx_y", self.Hx), ("Psi_Hx_z", self.Hx),
                            ("Psi_Hy_x", self.Hy), ("Psi_Hy_z", self.Hy),
                            ("Psi_Hz_x", self.Hz), ("Psi_Hz_y", self.Hz),
                            ("Psi_Ex_y", self.Ex), ("Psi_Ex_z", self.Ex),
                            ("Psi_Ey_x", self.Ey), ("Psi_Ey_z", self.Ey),
                            ("Psi_Ez_x", self.Ez), ("Psi_Ez_y", self.Ez)):
            setattr(self, name, np.zeros_like(field))

    def _pml_profile(self, coordinates, length, thickness, order, sigma_max,
                     kappa_max, alpha_max):
        depth = np.maximum(np.maximum(thickness - coordinates,
                                     coordinates - (length - thickness)), 0.0)
        u = np.clip(depth / thickness, 0.0, 1.0)
        sigma = sigma_max * u**order
        kappa = 1.0 + (kappa_max - 1.0) * u**order
        alpha = alpha_max * (1.0-u)
        alpha[u == 0.0] = 0.0
        b = np.exp(-(sigma / (self.eps0*kappa) + alpha/self.eps0) * self.dt)
        den = sigma*kappa + alpha*kappa*kappa
        c = np.zeros_like(b)
        valid = den != 0.0
        c[valid] = sigma[valid] / den[valid] * (b[valid]-1.0)
        return np.ascontiguousarray(kappa), np.ascontiguousarray(b), np.ascontiguousarray(c)

    def add_PML(self, pml_width, order=3, direction="xyz", sigma_max=None,
                kappa_max=7.0, alpha_max=0.05, R0=1e-8):
        """Configure matched convolutional PML on both ends of selected axes."""
        direction = str(direction).lower()
        if not direction or any(a not in "xyz" for a in direction):
            raise ValueError("direction must contain x, y and/or z.")
        if isinstance(pml_width, (int, np.integer)):
            widths = (int(pml_width),) * 3
        elif np.ndim(pml_width) == 0:
            widths = tuple(max(1, int(np.ceil(float(pml_width)/d))) for d in (self.dx,self.dy,self.dz))
        elif len(pml_width) == 3:
            widths = tuple(int(v) if isinstance(v, (int,np.integer)) else
                           max(1, int(np.ceil(float(v)/d))) for v,d in zip(pml_width,(self.dx,self.dy,self.dz)))
        else:
            raise TypeError("pml_width must be cells/metres or a length-three sequence.")
        if order < 1 or kappa_max < 1 or alpha_max < 0 or not (0 < R0 < 1):
            raise ValueError("Invalid CPML grading parameters.")
        self._set_identity_cpml()
        counts = (self.Nx, self.Ny, self.Nz)
        spacings = (self.dx, self.dy, self.dz)
        lengths = (self.x_range, self.y_range, self.z_range)
        actual = [0, 0, 0]
        for axis, name in enumerate("xyz"):
            if name not in direction:
                continue
            width = widths[axis]
            if not 1 <= width < counts[axis]//2:
                raise ValueError(f"PML width on {name} must be between 1 and {counts[axis]//2-1} cells.")
            actual[axis] = width
            thickness = width * spacings[axis]
            sm = (-(order+1)*np.log(R0)/(2*self.eta0*thickness)
                  if sigma_max is None else float(sigma_max))
            node = self._pml_profile(np.arange(counts[axis]+1)*spacings[axis],
                                     lengths[axis], thickness, order, sm, kappa_max, alpha_max)
            cell = self._pml_profile((np.arange(counts[axis])+0.5)*spacings[axis],
                                     lengths[axis], thickness, order, sm, kappa_max, alpha_max)
            self._pml[name] = dict(zip(("node_k","node_b","node_c","cell_k","cell_b","cell_c"), node+cell))
        self.pml_width = tuple(actual)
        return self

    add_pml = add_PML

    # -------------------------------------------------------------------- sources
    def _grid_index(self, value, axis, size, component):
        if isinstance(value, (int, np.integer)):
            return int(np.clip(value, 0, size-1))
        offset = 0.5 if axis == component else 0.0
        step = (self.dx, self.dy, self.dz)[axis]
        return int(np.clip(np.round(float(value)/step-offset), 0, size-1))

    def _source_span(self, value, axis, size, component):
        if isinstance(value, (tuple, list, np.ndarray)):
            if len(value) != 2:
                raise TypeError("Source spans must contain two values.")
            lo = self._grid_index(value[0], axis, size, component)
            if isinstance(value[1], (int, np.integer)):
                hi = int(np.clip(value[1], 0, size))
            else:
                step = (self.dx,self.dy,self.dz)[axis]
                offset = 0.5 if axis == component else 0.0
                hi = int(np.clip(np.round(float(value[1])/step-offset), 0, size))
            lo, hi = min(lo, hi), max(lo, hi)
            return lo, max(lo+1, hi), True
        i = self._grid_index(value, axis, size, component)
        return i, i+1, False

    def add_source(self, kind, x, y, z, amplitude=1.0, t0=None, tw=None,
                   f_min=None, f_max=None, polarization="z"):
        """Add a soft electric point, line or plane source on the Yee lattice."""
        kind = str(kind).lower()
        if kind not in {"point", "line", "plane"}:
            raise ValueError("kind must be 'point', 'line', or 'plane'.")
        pol = str(polarization).lower()
        if pol not in "xyz":
            raise ValueError("polarization must be x, y, or z.")
        component = "xyz".index(pol)
        shape = (self.Ex.shape, self.Ey.shape, self.Ez.shape)[component]
        spans = [self._source_span(v, a, shape[a], component)
                 for a,v in enumerate((x,y,z))]
        required = {"point": 0, "line": 1, "plane": 2}[kind]
        if sum(item[2] for item in spans) != required:
            raise ValueError(f"A {kind} source requires exactly {required} spatial span(s).")
        grids = np.meshgrid(*[np.arange(s[0],s[1],dtype=np.int32) for s in spans], indexing="ij")
        coords = np.ascontiguousarray(np.column_stack([g.ravel() for g in grids]), dtype=np.int32)
        fm = self.f_max if f_max is None else float(f_max)
        source = {
            "kind": kind, "pol": pol, "coords": coords,
            "amplitude": float(amplitude), "t0": 4.0/fm if t0 is None else float(t0),
            "tw": 1.0/fm if tw is None else float(tw),
            "f_min": self.f_min if f_min is None else float(f_min), "f_max": fm,
        }
        if source["tw"] <= 0 or source["f_max"] <= 0:
            raise ValueError("tw and f_max must be positive.")
        self.sources.append(source)
        return len(self.sources)-1

    def _g(self, source, t):
        t = np.asarray(t, dtype=float)
        if source["f_min"] is None:
            return source["amplitude"]*np.exp(-((t-source["t0"])/source["tw"])**2)
        if np.isclose(source["f_min"], source["f_max"]):
            f = source["f_max"]
            tau = np.maximum(t-source["t0"], 0.0)
            return source["amplitude"]*(1-np.exp(-(tau*max(f,1e-30))**3))*np.sin(2*np.pi*f*(t-source["t0"]))
        f = 0.5*(source["f_min"]+source["f_max"])
        return source["amplitude"]*np.sin(2*np.pi*f*(t-source["t0"]))*np.exp(-((t-source["t0"])/source["tw"])**2)

    # ------------------------------------------------------------------- monitors
    def add_plane_monitor(self, axis, position, first=None, second=None, index=None,
                          normal="+", save_path=None):
        """Add a cell-centered rectangular plane monitor.

        ``first`` and ``second`` are the two transverse half-open spans.  They
        default to the complete plane.  Integer values are cell indices and
        floating-point values are metres.
        """
        axis = str(axis).lower()
        if axis not in "xyz":
            raise ValueError("axis must be x, y, or z.")
        a = "xyz".index(axis)
        counts = (self.Nx,self.Ny,self.Nz)
        spacings = (self.dx,self.dy,self.dz)
        fixed = int(np.clip(position,0,counts[a]-1)) if isinstance(position,(int,np.integer)) else int(np.clip(np.floor(float(position)/spacings[a]),0,counts[a]-1))
        trans = [q for q in range(3) if q != a]
        def parse(span, q):
            if span is None:
                return 0, counts[q]
            if len(span) != 2:
                raise TypeError("Monitor spans must contain two values.")
            vals=[]
            for value in span:
                vals.append(int(value) if isinstance(value,(int,np.integer)) else int(np.floor(float(value)/spacings[q])))
            lo,hi=sorted(vals)
            return max(0,lo), min(counts[q],hi)
        spans = [parse(first,trans[0]), parse(second,trans[1])]
        if any(hi <= lo for lo,hi in spans):
            raise ValueError("Plane monitor spans must be non-empty.")
        used={m["index"] for m in self.monitors}
        if index is None:
            index=0
            while index in used: index+=1
        if not isinstance(index,(int,np.integer)) or index < 0 or int(index) in used:
            raise ValueError("index must be a unique non-negative integer.")
        sign = -1.0 if str(normal).strip().startswith("-") else 1.0
        n=np.zeros(3); n[a]=sign
        coords=[]
        for u in range(*spans[0]):
            for v in range(*spans[1]):
                item=[0,0,0]; item[a]=fixed; item[trans[0]]=u; item[trans[1]]=v; coords.append(item)
        coords=np.ascontiguousarray(coords,dtype=np.int32)
        xyz=(coords+0.5)*np.array(spacings)[None,:]
        monitor={"index":int(index),"axis":axis,"position":fixed,"normal":n,
                 "coords":coords,"positions":xyz,"shape":tuple(hi-lo for lo,hi in spans),
                 "spans":tuple(spans),"dA":spacings[trans[0]]*spacings[trans[1]],
                 "transverse_axes":"".join("xyz"[q] for q in trans),
                 "extent":(spans[0][0]*spacings[trans[0]],spans[0][1]*spacings[trans[0]],
                           spans[1][0]*spacings[trans[1]],spans[1][1]*spacings[trans[1]]),
                 "save_path":None if save_path is None else str(save_path)}
        self.monitors.append(monitor)
        return int(index)

    def add_nf2ff_box(self, x, y, z, start_index=None):
        """Create six outward-facing plane monitors and return their IDs."""
        spans=[]
        for span,count in zip((x,y,z),(self.Nx,self.Ny,self.Nz)):
            if len(span)!=2: raise TypeError("Box spans must have two cell indices.")
            lo,hi=sorted(map(int,span))
            if not (0 <= lo < hi <= count): raise ValueError("Invalid NF2FF box span.")
            spans.append((lo,hi))
        next_id = start_index
        result={}
        specs=(("x_min","x",spans[0][0],spans[1],spans[2],"-"),
               ("x_max","x",spans[0][1]-1,spans[1],spans[2],"+"),
               ("y_min","y",spans[1][0],spans[0],spans[2],"-"),
               ("y_max","y",spans[1][1]-1,spans[0],spans[2],"+"),
               ("z_min","z",spans[2][0],spans[0],spans[1],"-"),
               ("z_max","z",spans[2][1]-1,spans[0],spans[1],"+"))
        for key,axis,pos,one,two,normal in specs:
            result[key]=self.add_plane_monitor(axis,pos,one,two,next_id,normal)
            if next_id is not None: next_id+=1
        return result

    def _monitor_by_id(self, index, results=True):
        collection=self.monitor_results if results else self.monitors
        for item in collection:
            if item["index"] == int(index): return item
        available=[m["index"] for m in collection]
        raise KeyError(f"Monitor {index} not found; available IDs are {available}.")

    # ---------------------------------------------------------------- simulation
    def reset_fields(self):
        for name in ("Ex","Ey","Ez","Hx","Hy","Hz"):
            getattr(self,name).fill(0.0)
        self._allocate_psi()
        self.current_step=0
        self.monitor_results=[]
        return self

    def _compile_sources(self, steps):
        t=(self.current_step+np.arange(steps)+1)*self.dt
        values=np.empty((steps,len(self.sources)))
        coords=[]; ids=[]; pols=[]
        masks=(self.PEC_Ex,self.PEC_Ey,self.PEC_Ez)
        for sid,source in enumerate(self.sources):
            values[:,sid]=self._g(source,t)
            pol="xyz".index(source["pol"])
            c=source["coords"]
            if np.any(masks[pol][c[:,0],c[:,1],c[:,2]]):
                raise ValueError(f"Source {sid} intersects a PEC object or terminating boundary.")
            coords.append(c); ids.extend([sid]*len(c)); pols.extend([pol]*len(c))
        allcoords=np.vstack(coords) if coords else np.empty((0,3),dtype=np.int32)
        return (np.ascontiguousarray(values),np.ascontiguousarray(allcoords,dtype=np.int32),
                np.ascontiguousarray(ids,dtype=np.int32),np.ascontiguousarray(pols,dtype=np.int8))

    def _compile_monitors(self, steps, stride):
        coords=np.vstack([m["coords"] for m in self.monitors]) if self.monitors else np.empty((0,3),dtype=np.int32)
        record_steps=np.arange(0,steps,stride,dtype=np.int64)
        history=np.empty((len(record_steps),len(coords),6),dtype=float)
        return np.ascontiguousarray(coords,dtype=np.int32),history,record_steps

    def _kernel_arguments(self, source_data, monitor_coords, history, stride):
        p=self._pml
        arrays=[self.Ex,self.Ey,self.Ez,self.Hx,self.Hy,self.Hz,
                self.Psi_Hx_y,self.Psi_Hx_z,self.Psi_Hy_x,self.Psi_Hy_z,self.Psi_Hz_x,self.Psi_Hz_y,
                self.Psi_Ex_y,self.Psi_Ex_z,self.Psi_Ey_x,self.Psi_Ey_z,self.Psi_Ez_x,self.Psi_Ez_y,
                self.CaEx,self.CbEx,self.CaEy,self.CbEy,self.CaEz,self.CbEz,
                self.CaHx,self.CbHx,self.CaHy,self.CbHy,self.CaHz,self.CbHz]
        cpml=[]
        for name in "xyz": cpml.extend([p[name]["node_k"],p[name]["node_b"],p[name]["node_c"],p[name]["cell_k"],p[name]["cell_b"],p[name]["cell_c"]])
        return arrays+cpml+[self.dx,self.dy,self.dz,*source_data,monitor_coords,history,int(stride)]

    def run(self, steps=None, record_stride=1, reset=False, progress=True,
            progress_desc="3D FDTD"):
        """Advance the simulation, with a tqdm progress bar by default."""
        if reset: self.reset_fields()
        steps=self.Nt if steps is None else int(steps)
        stride=int(record_stride)
        if steps < 1 or stride < 1: raise ValueError("steps and record_stride must be positive.")
        source_data=self._compile_sources(steps)
        monitor_coords,history,record_steps=self._compile_monitors(steps,stride)
        if progress:
            try:
                from tqdm.auto import tqdm
            except ImportError as exc:
                raise ImportError("Simulation progress display requires tqdm.") from exc
            progress_bar=tqdm(total=steps,desc=str(progress_desc),unit="step",dynamic_ncols=True)
            target_chunks=100
            base=max(1,int(np.ceil(steps/target_chunks)))
            chunk_steps=max(stride,int(np.ceil(base/stride))*stride)
        else:
            progress_bar=None; chunk_steps=steps
        try:
            for start in range(0,steps,chunk_steps):
                stop=min(start+chunk_steps,steps); count=stop-start
                values,coords,source_ids,pols=source_data
                chunk_source=(np.ascontiguousarray(values[start:stop]),coords,source_ids,pols)
                rec_start=start//stride
                rec_count=(count+stride-1)//stride
                chunk_history=history[rec_start:rec_start+rec_count]
                if self.backend == "cython":
                    _cython_kernel.run_fdtd(*self._kernel_arguments(
                        chunk_source,monitor_coords,chunk_history,stride))
                else:
                    self._run_numpy(count,chunk_source,monitor_coords,chunk_history,stride)
                if progress_bar is not None: progress_bar.update(count)
        finally:
            if progress_bar is not None: progress_bar.close()
        times=(self.current_step+record_steps+1)*self.dt
        self.current_step += steps
        self.monitor_results=[]; offset=0
        for monitor in self.monitors:
            count=len(monitor["coords"])
            item={**monitor,"dt":self.dt,"time":times.copy(),"fields":history[:,offset:offset+count,:].reshape((len(times),)+monitor["shape"]+(6,))}
            self.monitor_results.append(item); offset+=count
        self.last_source_time=(self.current_step-steps+np.arange(steps)+1)*self.dt
        self.last_source_waveforms=source_data[0]
        for monitor in self.monitor_results:
            if monitor.get("save_path") is not None:
                self._write_plane_monitor(monitor, monitor["save_path"])
        return self.monitor_results

    def step(self, n=None):
        """Advance one time step (``n`` is accepted for legacy call sites)."""
        self.run(steps=1,progress=False)
        return self

    def _run_numpy(self, steps, source_data, monitor_coords, history, stride):
        values,coords,source_ids,pols=source_data
        rec=0
        for n in range(steps):
            self._update_h_numpy(); self._update_e_numpy()
            for q,(i,j,k) in enumerate(coords):
                (self.Ex,self.Ey,self.Ez)[pols[q]][i,j,k] += values[n,source_ids[q]]
            if n % stride == 0:
                history[rec]=self._sample_cells(monitor_coords); rec+=1

    def _update_h_numpy(self):
        p=self._pml
        dzy=(self.Ez[:,1:,:]-self.Ez[:,:-1,:])/self.dy
        dyz=(self.Ey[:,:,1:]-self.Ey[:,:,:-1])/self.dz
        self.Psi_Hx_y[:]=p["y"]["cell_b"][None,:,None]*self.Psi_Hx_y+p["y"]["cell_c"][None,:,None]*dzy
        self.Psi_Hx_z[:]=p["z"]["cell_b"][None,None,:]*self.Psi_Hx_z+p["z"]["cell_c"][None,None,:]*dyz
        curl=dzy/p["y"]["cell_k"][None,:,None]+self.Psi_Hx_y-dyz/p["z"]["cell_k"][None,None,:]-self.Psi_Hx_z
        self.Hx[:]=self.CaHx*self.Hx-self.CbHx*curl
        dxz=(self.Ex[:,:,1:]-self.Ex[:,:,:-1])/self.dz
        dzx=(self.Ez[1:,:,:]-self.Ez[:-1,:,:])/self.dx
        self.Psi_Hy_z[:]=p["z"]["cell_b"][None,None,:]*self.Psi_Hy_z+p["z"]["cell_c"][None,None,:]*dxz
        self.Psi_Hy_x[:]=p["x"]["cell_b"][:,None,None]*self.Psi_Hy_x+p["x"]["cell_c"][:,None,None]*dzx
        curl=dxz/p["z"]["cell_k"][None,None,:]+self.Psi_Hy_z-dzx/p["x"]["cell_k"][:,None,None]-self.Psi_Hy_x
        self.Hy[:]=self.CaHy*self.Hy-self.CbHy*curl
        dyx=(self.Ey[1:,:,:]-self.Ey[:-1,:,:])/self.dx
        dxy=(self.Ex[:,1:,:]-self.Ex[:,:-1,:])/self.dy
        self.Psi_Hz_x[:]=p["x"]["cell_b"][:,None,None]*self.Psi_Hz_x+p["x"]["cell_c"][:,None,None]*dyx
        self.Psi_Hz_y[:]=p["y"]["cell_b"][None,:,None]*self.Psi_Hz_y+p["y"]["cell_c"][None,:,None]*dxy
        curl=dyx/p["x"]["cell_k"][:,None,None]+self.Psi_Hz_x-dxy/p["y"]["cell_k"][None,:,None]-self.Psi_Hz_y
        self.Hz[:]=self.CaHz*self.Hz-self.CbHz*curl

    def _update_e_numpy(self):
        p=self._pml
        dzy=(self.Hz[:,1:,:]-self.Hz[:,:-1,:])/self.dy
        dyz=(self.Hy[:,:,1:]-self.Hy[:,:,:-1])/self.dz
        self.Psi_Ex_y[:,1:-1,1:-1]=p["y"]["node_b"][None,1:-1,None]*self.Psi_Ex_y[:,1:-1,1:-1]+p["y"]["node_c"][None,1:-1,None]*dzy[:,:,1:-1]
        self.Psi_Ex_z[:,1:-1,1:-1]=p["z"]["node_b"][None,None,1:-1]*self.Psi_Ex_z[:,1:-1,1:-1]+p["z"]["node_c"][None,None,1:-1]*dyz[:,1:-1,:]
        curl=dzy[:,:,1:-1]/p["y"]["node_k"][None,1:-1,None]+self.Psi_Ex_y[:,1:-1,1:-1]-dyz[:,1:-1,:]/p["z"]["node_k"][None,None,1:-1]-self.Psi_Ex_z[:,1:-1,1:-1]
        self.Ex[:,1:-1,1:-1]=self.CaEx[:,1:-1,1:-1]*self.Ex[:,1:-1,1:-1]+self.CbEx[:,1:-1,1:-1]*curl
        dxz=(self.Hx[:,:,1:]-self.Hx[:,:,:-1])/self.dz
        dzx=(self.Hz[1:,:,:]-self.Hz[:-1,:,:])/self.dx
        self.Psi_Ey_z[1:-1,:,1:-1]=p["z"]["node_b"][None,None,1:-1]*self.Psi_Ey_z[1:-1,:,1:-1]+p["z"]["node_c"][None,None,1:-1]*dxz[1:-1,:,:]
        self.Psi_Ey_x[1:-1,:,1:-1]=p["x"]["node_b"][1:-1,None,None]*self.Psi_Ey_x[1:-1,:,1:-1]+p["x"]["node_c"][1:-1,None,None]*dzx[:,:,1:-1]
        curl=dxz[1:-1,:,:]/p["z"]["node_k"][None,None,1:-1]+self.Psi_Ey_z[1:-1,:,1:-1]-dzx[:,:,1:-1]/p["x"]["node_k"][1:-1,None,None]-self.Psi_Ey_x[1:-1,:,1:-1]
        self.Ey[1:-1,:,1:-1]=self.CaEy[1:-1,:,1:-1]*self.Ey[1:-1,:,1:-1]+self.CbEy[1:-1,:,1:-1]*curl
        dyx=(self.Hy[1:,:,:]-self.Hy[:-1,:,:])/self.dx
        dxy=(self.Hx[:,1:,:]-self.Hx[:,:-1,:])/self.dy
        self.Psi_Ez_x[1:-1,1:-1,:]=p["x"]["node_b"][1:-1,None,None]*self.Psi_Ez_x[1:-1,1:-1,:]+p["x"]["node_c"][1:-1,None,None]*dyx[:,1:-1,:]
        self.Psi_Ez_y[1:-1,1:-1,:]=p["y"]["node_b"][None,1:-1,None]*self.Psi_Ez_y[1:-1,1:-1,:]+p["y"]["node_c"][None,1:-1,None]*dxy[1:-1,:,:]
        curl=dyx[:,1:-1,:]/p["x"]["node_k"][1:-1,None,None]+self.Psi_Ez_x[1:-1,1:-1,:]-dxy[1:-1,:,:]/p["y"]["node_k"][None,1:-1,None]-self.Psi_Ez_y[1:-1,1:-1,:]
        self.Ez[1:-1,1:-1,:]=self.CaEz[1:-1,1:-1,:]*self.Ez[1:-1,1:-1,:]+self.CbEz[1:-1,1:-1,:]*curl

    def _sample_cells(self, coords):
        if len(coords)==0: return np.empty((0,6))
        i,j,k=coords.T
        out=np.empty((len(coords),6))
        out[:,0]=0.25*(self.Ex[i,j,k]+self.Ex[i,j+1,k]+self.Ex[i,j,k+1]+self.Ex[i,j+1,k+1])
        out[:,1]=0.25*(self.Ey[i,j,k]+self.Ey[i+1,j,k]+self.Ey[i,j,k+1]+self.Ey[i+1,j,k+1])
        out[:,2]=0.25*(self.Ez[i,j,k]+self.Ez[i+1,j,k]+self.Ez[i,j+1,k]+self.Ez[i+1,j+1,k])
        out[:,3]=0.5*(self.Hx[i,j,k]+self.Hx[i+1,j,k])
        out[:,4]=0.5*(self.Hy[i,j,k]+self.Hy[i,j+1,k])
        out[:,5]=0.5*(self.Hz[i,j,k]+self.Hz[i,j,k+1])
        return out

    # ----------------------------------------------------------- monitor storage
    @staticmethod
    def _monitor_file_path(path):
        target=Path(path).expanduser()
        if target.suffix.lower() not in {".h5",".hdf5"}:
            target=Path(str(target)+".h5")
        return target

    @staticmethod
    def _h5py():
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("Plane-monitor HDF5 storage requires h5py.") from exc
        return h5py

    def _write_plane_monitor(self, monitor, path):
        target=self._monitor_file_path(path)
        target.parent.mkdir(parents=True,exist_ok=True)
        h5py=self._h5py()
        with h5py.File(target,"w") as handle:
            handle.attrs["format"]="FDTD_3D_plane_monitor"
            handle.attrs["format_version"]=1
            handle.attrs["index"]=int(monitor["index"])
            handle.attrs["axis"]=monitor["axis"]
            handle.attrs["position"]=int(monitor["position"])
            handle.attrs["dA"]=float(monitor["dA"])
            handle.attrs["transverse_axes"]=monitor["transverse_axes"]
            handle.attrs["dt"]=float(monitor.get("dt",self.dt))
            handle.attrs["field_components"]="Ex,Ey,Ez,Hx,Hy,Hz"
            handle.create_dataset("normal",data=np.asarray(monitor["normal"],dtype=float))
            handle.create_dataset("coords",data=np.asarray(monitor["coords"],dtype=np.int32),
                                  compression="gzip",compression_opts=4,shuffle=True)
            handle.create_dataset("positions",data=np.asarray(monitor["positions"],dtype=float),
                                  compression="gzip",compression_opts=4,shuffle=True)
            handle.create_dataset("shape",data=np.asarray(monitor["shape"],dtype=np.int64))
            handle.create_dataset("spans",data=np.asarray(monitor["spans"],dtype=np.int64))
            handle.create_dataset("extent",data=np.asarray(monitor["extent"],dtype=float))
            handle.create_dataset("time",data=np.asarray(monitor["time"],dtype=float),
                                  compression="gzip",compression_opts=4,shuffle=True)
            handle.create_dataset("fields",data=np.asarray(monitor["fields"],dtype=float),
                                  chunks=True,compression="gzip",compression_opts=4,shuffle=True)
        monitor["save_path"]=str(target.resolve())
        return target.resolve()

    def save_plane_monitor(self, monitor_index, path=None):
        """Save one recorded plane monitor to a compressed, chunked HDF5 file."""
        monitor=self._monitor_by_id(monitor_index)
        destination=monitor.get("save_path") if path is None else path
        if destination is None:
            raise ValueError("Provide path or configure save_path in add_plane_monitor().")
        return self._write_plane_monitor(monitor,destination)

    def load_plane_monitor(self, path, index=None, register=True):
        """Load saved plane data, optionally registering it for analysis/plotting."""
        source=self._monitor_file_path(path)
        h5py=self._h5py()
        required_datasets={"normal","coords","positions","shape","spans","extent","time","fields"}
        required_attrs={"index","axis","position","dA","transverse_axes","dt"}
        with h5py.File(source,"r") as data:
            missing=required_datasets.difference(data.keys())
            missing_attrs=required_attrs.difference(data.attrs.keys())
            if missing or missing_attrs:
                names=sorted(missing | {f"attribute:{name}" for name in missing_attrs})
                raise ValueError(f"Invalid plane-monitor HDF5 file; missing {names}.")
            if data.attrs.get("format","") != "FDTD_3D_plane_monitor":
                raise ValueError("HDF5 file is not an FDTD_3D plane monitor.")
            fields=np.asarray(data["fields"][...],dtype=float)
            time=np.asarray(data["time"][...],dtype=float)
            shape=tuple(int(v) for v in data["shape"][...])
            if fields.shape != (len(time),)+shape+(6,):
                raise ValueError("Saved plane-monitor field dimensions are inconsistent.")
            monitor={
                "index":int(data.attrs["index"]) if index is None else int(index),
                "axis":str(data.attrs["axis"]),"position":int(data.attrs["position"]),
                "normal":np.asarray(data["normal"][...],dtype=float),
                "coords":np.asarray(data["coords"][...],dtype=np.int32),
                "positions":np.asarray(data["positions"][...],dtype=float),"shape":shape,
                "spans":tuple(tuple(int(q) for q in row) for row in data["spans"][...]),
                "dA":float(data.attrs["dA"]),
                "transverse_axes":str(data.attrs["transverse_axes"]),
                "extent":tuple(float(v) for v in data["extent"][...]),
                "dt":float(data.attrs["dt"]),"time":time,"fields":fields,
                "save_path":str(source.resolve()),
            }
        if monitor["index"] < 0:
            raise ValueError("Loaded monitor index must be non-negative.")
        if register:
            if any(item["index"] == monitor["index"] for item in self.monitor_results):
                raise ValueError(f"Monitor index {monitor['index']} is already registered; provide a different index.")
            self.monitor_results.append(monitor)
        return monitor

    # -------------------------------------------------------------- postprocess
    def _frequencies(self, freqs):
        f=np.atleast_1d(np.asarray(freqs,dtype=float))
        if f.ndim!=1 or len(f)==0 or np.any(~np.isfinite(f)) or np.any(f<0):
            raise ValueError("freqs must be finite, non-negative scalar or 1D data.")
        return f

    def _dft(self, monitor, freqs, window=None, detrend=True):
        f=self._frequencies(freqs); data=monitor["fields"].reshape(len(monitor["time"]),-1,6)
        if len(data)<2: raise ValueError("At least two monitor samples are required.")
        if detrend: data=data-data.mean(axis=0,keepdims=True)
        if window is None: w=np.ones(len(data))
        elif str(window).lower() in {"hann","hanning"}: w=np.hanning(len(data))
        elif str(window).lower()=="hamming": w=np.hamming(len(data))
        elif str(window).lower()=="blackman": w=np.blackman(len(data))
        else: raise ValueError("window must be None, hann, hamming, or blackman.")
        coherent_gain=max(float(np.sum(w)),1e-30)
        kernel=np.exp(-2j*np.pi*f[:,None]*monitor["time"][None,:])*w[None,:]/coherent_gain
        result=np.einsum("ft,tpq->fpq",kernel,data)
        result[:,:,3:]*=np.exp(1j*2*np.pi*f[:,None,None]*monitor.get("dt",self.dt)/2)
        return result

    def _source_dft(self, index, freqs, window=None, detrend=True):
        if not hasattr(self,"last_source_waveforms"): raise RuntimeError("Run the simulation first.")
        if not 0 <= int(index) < self.last_source_waveforms.shape[1]: raise IndexError("source_index out of range.")
        f=self._frequencies(freqs); values=self.last_source_waveforms[:,int(index)].copy()
        if detrend: values-=values.mean()
        if window is None: w=np.ones(len(values))
        elif str(window).lower() in {"hann","hanning"}: w=np.hanning(len(values))
        elif str(window).lower()=="hamming": w=np.hamming(len(values))
        elif str(window).lower()=="blackman": w=np.blackman(len(values))
        else: raise ValueError("window must be None, hann, hamming, or blackman.")
        kernel=np.exp(-2j*np.pi*f[:,None]*self.last_source_time[None,:])*w[None,:]/max(float(w.sum()),1e-30)
        return kernel@values

    def power_spectrum(self, monitor_index, freqs, source_index=None, window=None, detrend=True):
        """Integrate signed time-average Poynting flux over a plane monitor."""
        monitor=self._monitor_by_id(monitor_index)
        fields=self._dft(monitor,freqs,window,detrend)
        flux=0.5*np.einsum("fpi,fpi->fp",np.cross(fields[:,:,:3],np.conj(fields[:,:,3:])),monitor["normal"][None,None,:])
        raw=np.sum(flux,axis=1)*monitor["dA"]
        result={"freqs":self._frequencies(freqs),"raw_complex_power":raw,"raw_power":raw.real,
                "monitor_index":int(monitor_index),"normal":monitor["normal"].copy(),"normalized":False}
        if source_index is None:
            result["power"]=raw.real; result["complex_power"]=raw
        else:
            source=self._source_dft(source_index,freqs,window,detrend); source_power=np.abs(source)**2
            valid=source_power >= 1e-12*max(float(source_power.max()),1e-30)
            normalized=np.zeros_like(raw); normalized[valid]=raw[valid]/source_power[valid]
            result.update(power=normalized.real,complex_power=normalized,source_power=source_power,
                          source_index=int(source_index),valid_source=valid,normalized=True)
        return result

    def NF2FF(self, surfaces, freqs, theta=None, phi=None, r_obs=1.0,
              source_index=None, window=None, detrend=True):
        """Apply the 3D surface-equivalence near-field to far-field transform."""
        if isinstance(surfaces,dict): ids=list(surfaces.values())
        else: ids=list(surfaces)
        if not ids: raise ValueError("At least one plane monitor is required.")
        f=self._frequencies(freqs)
        theta=np.linspace(0,np.pi,181) if theta is None else np.atleast_1d(theta).astype(float)
        phi=np.linspace(0,2*np.pi,361,endpoint=False) if phi is None else np.atleast_1d(phi).astype(float)
        if r_obs <= 0: raise ValueError("r_obs must be positive.")
        TH,PH=np.meshgrid(theta,phi,indexing="ij")
        rhat=np.stack((np.sin(TH)*np.cos(PH),np.sin(TH)*np.sin(PH),np.cos(TH)),axis=-1).reshape(-1,3)
        that=np.stack((np.cos(TH)*np.cos(PH),np.cos(TH)*np.sin(PH),-np.sin(TH)),axis=-1).reshape(-1,3)
        phat=np.stack((-np.sin(PH),np.cos(PH),np.zeros_like(PH)),axis=-1).reshape(-1,3)
        Eth=np.zeros((len(f),len(rhat)),complex); Eph=np.zeros_like(Eth)
        monitor_data=[]
        for mid in ids:
            m=self._monitor_by_id(mid); monitor_data.append((m,self._dft(m,f,window,detrend)))
        for fi,freq in enumerate(f):
            k0=2*np.pi*freq/self.c0
            N=np.zeros((len(rhat),3),complex); L=np.zeros_like(N)
            for monitor,fields in monitor_data:
                E=fields[fi,:,:3]; H=fields[fi,:,3:]; n=monitor["normal"]
                J=np.cross(n[None,:],H); M=-np.cross(n[None,:],E)
                for start in range(0,len(rhat),512):
                    stop=min(start+512,len(rhat))
                    phase=np.exp(-1j*k0*(rhat[start:stop]@monitor["positions"].T))
                    N[start:stop]+=phase@J*monitor["dA"]
                    L[start:stop]+=phase@M*monitor["dA"]
            Nt=np.sum(N*that,axis=1); Np=np.sum(N*phat,axis=1)
            Lt=np.sum(L*that,axis=1); Lp=np.sum(L*phat,axis=1)
            pref=1j*k0*np.exp(1j*k0*r_obs)/(4*np.pi*r_obs)
            Eth[fi]=pref*(self.eta0*Nt+Lp)
            Eph[fi]=pref*(self.eta0*Np-Lt)
        if source_index is not None:
            spectrum=self._source_dft(source_index,f,window,detrend)
            valid=np.abs(spectrum)>=1e-12*max(float(np.abs(spectrum).max()),1e-30)
            Eth[valid]/=spectrum[valid,None]; Eph[valid]/=spectrum[valid,None]
            Eth[~valid]=0; Eph[~valid]=0
        shape=(len(f),len(theta),len(phi)); Eth=Eth.reshape(shape); Eph=Eph.reshape(shape)
        radiation=r_obs*r_obs*(np.abs(Eth)**2+np.abs(Eph)**2)/(2*self.eta0)
        return {"freqs":f,"theta":theta,"phi":phi,"Etheta":Eth,"Ephi":Eph,
                "radiation_intensity":radiation,"r_obs":float(r_obs),"surface_monitors":ids,
                "source_index":source_index}

    nf2ff = NF2FF

    def plot_power_spectrum(self, spectrum, db=False, ax=None):
        import matplotlib.pyplot as plt
        if ax is None: _,ax=plt.subplots(figsize=(7,4))
        values=np.asarray(spectrum["power"])
        plotted=10*np.log10(np.maximum(np.abs(values),1e-30)) if db else values
        ax.plot(np.asarray(spectrum["freqs"])/1e9,plotted)
        ax.set(xlabel="Frequency (GHz)",ylabel=("Power (dB)" if db else ("Source-normalized power" if spectrum["normalized"] else "Power (W)")),title=f"Plane power: monitor {spectrum['monitor_index']}")
        ax.grid(True,alpha=.3); ax.figure.tight_layout(); return ax.figure,ax

    def plot_nf2ff(self, result, frequency_index=0, db=True, db_floor=-40.0,
                   cmap="viridis", ax=None):
        """Plot the far-field radiation pattern as a 3D dB-scaled surface.

        With ``db=True`` (the default), color is normalized intensity in dB and
        radius is the clipped dB value shifted so ``db_floor`` maps to zero.
        """
        import matplotlib.pyplot as plt
        from matplotlib import colors, cm
        theta=np.asarray(result["theta"],dtype=float)
        phi=np.asarray(result["phi"],dtype=float)
        intensity=np.asarray(result["radiation_intensity"],dtype=float)
        fi=int(frequency_index)
        if intensity.ndim != 3 or intensity.shape[1:] != (len(theta),len(phi)):
            raise ValueError("NF2FF radiation-intensity dimensions are inconsistent.")
        if not 0 <= fi < intensity.shape[0]:
            raise IndexError("frequency_index is outside the NF2FF result.")
        if len(theta)<2 or len(phi)<3:
            raise ValueError("A 3D far-field plot requires at least 2 theta and 3 phi samples.")
        values=np.maximum(intensity[fi],0.0)
        # Close the azimuth seam when phi was sampled with endpoint=False.
        if not np.isclose((phi[-1]-phi[0])%(2*np.pi),0.0,atol=1e-10):
            phi_plot=np.concatenate((phi,[phi[0]+2*np.pi]))
            values=np.concatenate((values,values[:,:1]),axis=1)
        else:
            phi_plot=phi
        peak=max(float(np.nanmax(values)),1e-300)
        normalized=values/peak
        if db:
            floor=float(db_floor)
            if not np.isfinite(floor) or floor >= 0.0:
                raise ValueError("db_floor must be a finite negative value.")
            color_values=np.maximum(10*np.log10(np.maximum(normalized,1e-300)),floor)
            radius=(color_values-floor)/(-floor)
            norm=colors.Normalize(vmin=floor,vmax=0.0)
            colorbar_label="Normalized radiation intensity (dB)"
        else:
            color_values=normalized; radius=normalized
            norm=colors.Normalize(vmin=0.0,vmax=1.0)
            colorbar_label="Normalized radiation intensity"
        TH,PH=np.meshgrid(theta,phi_plot,indexing="ij")
        X=radius*np.sin(TH)*np.cos(PH)
        Y=radius*np.sin(TH)*np.sin(PH)
        Z=radius*np.cos(TH)
        if ax is None:
            figure=plt.figure(figsize=(8,7)); ax=figure.add_subplot(111,projection="3d")
        else:
            figure=ax.figure
            if getattr(ax,"name","") != "3d":
                raise TypeError("ax must be a 3D Matplotlib axes.")
        colormap=plt.get_cmap(cmap)
        ax.plot_surface(X,Y,Z,facecolors=colormap(norm(color_values)),
                        linewidth=0,antialiased=True,shade=False)
        mapper=cm.ScalarMappable(norm=norm,cmap=colormap); mapper.set_array(color_values)
        figure.colorbar(mapper,ax=ax,shrink=0.72,pad=0.08,label=colorbar_label)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_box_aspect((1,1,1))
        scale="dB" if db else "linear"
        ax.set_title(f"3D NF2FF at {result['freqs'][fi]/1e9:.4g} GHz ({scale} scale)")
        figure.tight_layout(); return figure,ax

    def plot_nf2ff_cut(self, result, frequency_index=0, theta_index=None,
                       db=True, db_floor=-40.0, ax=None):
        """Plot an optional azimuthal polar cut through an NF2FF result."""
        import matplotlib.pyplot as plt
        theta=np.asarray(result["theta"]); phi=np.asarray(result["phi"])
        ti=int(np.argmin(np.abs(theta-np.pi/2))) if theta_index is None else int(theta_index)
        values=np.asarray(result["radiation_intensity"])[frequency_index,ti]
        if db:
            values=10*np.log10(np.maximum(values/max(float(values.max()),1e-300),1e-300))
            values=np.maximum(values,float(db_floor))
        if ax is None: _,ax=plt.subplots(subplot_kw={"projection":"polar"})
        ax.plot(phi,values)
        ax.set_title(f"NF2FF cut at {result['freqs'][frequency_index]/1e9:.4g} GHz, theta={np.degrees(theta[ti]):.1f} deg")
        if db: ax.set_ylabel("Normalized intensity (dB)")
        ax.figure.tight_layout(); return ax.figure,ax

    def plot_plane_monitor(self, monitor, component="Ez", time_index=-1,
                           frequency=None, representation=None, window=None,
                           detrend=True, cmap=None, ax=None):
        """Plot recorded or reloaded plane-monitor data.

        ``monitor`` may be a registered monitor ID, a monitor dictionary, or an
        HDF5 path created by :meth:`save_plane_monitor`. At a requested
        ``frequency``, ``representation`` may be ``magnitude``, ``phase``,
        ``real`` or ``imag``. ``E``, ``H`` and ``Snormal`` are also accepted as
        derived component names.
        """
        import matplotlib.pyplot as plt
        if isinstance(monitor,(str,Path)):
            data=self.load_plane_monitor(monitor,register=False)
        elif isinstance(monitor,(int,np.integer)):
            data=self._monitor_by_id(int(monitor))
        elif isinstance(monitor,dict) and "fields" in monitor:
            data=monitor
        else:
            raise TypeError("monitor must be an ID, loaded monitor dictionary, or HDF5 path.")
        key=str(component).lower()
        components={"ex":0,"ey":1,"ez":2,"hx":3,"hy":4,"hz":5}
        if key not in {*components,"e","h","snormal"}:
            raise ValueError("component must be Ex/Ey/Ez/Hx/Hy/Hz, E, H, or Snormal.")
        if frequency is None:
            ti=int(time_index)
            if not -len(data["time"]) <= ti < len(data["time"]):
                raise IndexError("time_index is outside the recorded monitor history.")
            sample=np.asarray(data["fields"])[ti]
            values=self._plane_time_component(sample,key,data["normal"])
            default_rep="real"; detail=f"t={data['time'][ti]:.6g} s"
        else:
            freq=float(frequency)
            phasor=self._dft(data,[freq],window,detrend)[0].reshape(data["shape"]+(6,))
            if key in components: values=phasor[...,components[key]]
            elif key=="e": values=np.sqrt(np.sum(np.abs(phasor[...,:3])**2,axis=-1))
            elif key=="h": values=np.sqrt(np.sum(np.abs(phasor[...,3:])**2,axis=-1))
            else: values=0.5*np.real(np.einsum("...i,i->...",np.cross(phasor[...,:3],np.conj(phasor[...,3:])),data["normal"]))
            default_rep="magnitude"; detail=f"f={freq/1e9:.6g} GHz"
        rep=default_rep if representation is None else str(representation).lower()
        if np.iscomplexobj(values):
            if rep in {"magnitude","abs"}: plotted=np.abs(values); rep="magnitude"
            elif rep=="phase": plotted=np.angle(values)
            elif rep=="real": plotted=np.real(values)
            elif rep in {"imag","imaginary"}: plotted=np.imag(values); rep="imaginary"
            else: raise ValueError("representation must be magnitude, phase, real, or imag.")
        else:
            if rep not in {"real","magnitude","abs"}:
                raise ValueError("Only real/magnitude representation applies to this data.")
            plotted=np.abs(values) if rep in {"magnitude","abs"} else values
        if ax is None: _,ax=plt.subplots(figsize=(6,5))
        selected_cmap=cmap or ("RdBu_r" if rep in {"real","imaginary"} else
                               "twilight" if rep=="phase" else "viridis")
        image=ax.imshow(np.asarray(plotted).T,origin="lower",extent=data["extent"],
                        aspect="auto",cmap=selected_cmap)
        axes=data["transverse_axes"]
        ax.set_xlabel(f"{axes[0]} (m)"); ax.set_ylabel(f"{axes[1]} (m)")
        ax.set_title(f"Monitor {data['index']}: {component} {rep}, {detail}")
        ax.figure.colorbar(image,ax=ax,label=f"{component} ({rep})")
        ax.figure.tight_layout(); return ax.figure,ax

    plot_monitor = plot_plane_monitor

    @staticmethod
    def _plane_time_component(sample, component, normal):
        components={"ex":0,"ey":1,"ez":2,"hx":3,"hy":4,"hz":5}
        if component in components:
            return np.asarray(sample)[...,components[component]]
        if component=="e":
            return np.linalg.norm(np.asarray(sample)[...,:3],axis=-1)
        if component=="h":
            return np.linalg.norm(np.asarray(sample)[...,3:],axis=-1)
        return np.einsum("...i,i->...",np.cross(np.asarray(sample)[...,:3],
                                                np.asarray(sample)[...,3:]),normal)

    def animate_plane_monitor(self, monitor, component="Ez", frame_stride=1,
                              interval=50, repeat=True, vmin=None, vmax=None,
                              cmap=None, ax=None, save_path=None, fps=None, dpi=100):
        """Animate a recorded plane field, streaming frames from HDF5 when possible.

        ``monitor`` accepts the same ID/dictionary/path forms as
        :meth:`plot_plane_monitor`. ``save_path`` may be a GIF (Pillow) or a
        video format supported by the local Matplotlib installation.
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        key=str(component).lower()
        if key not in {"ex","ey","ez","hx","hy","hz","e","h","snormal"}:
            raise ValueError("component must be Ex/Ey/Ez/Hx/Hy/Hz, E, H, or Snormal.")
        stride=int(frame_stride)
        if stride < 1 or float(interval) <= 0:
            raise ValueError("frame_stride and interval must be positive.")
        h5_handle=None
        if isinstance(monitor,(str,Path)):
            source=self._monitor_file_path(monitor)
            h5py=self._h5py(); h5_handle=h5py.File(source,"r")
            try:
                if h5_handle.attrs.get("format","") != "FDTD_3D_plane_monitor":
                    raise ValueError("HDF5 file is not an FDTD_3D plane monitor.")
                fields=h5_handle["fields"]
                times=np.asarray(h5_handle["time"][...],dtype=float)
                normal=np.asarray(h5_handle["normal"][...],dtype=float)
                extent=tuple(float(v) for v in h5_handle["extent"][...])
                axes=str(h5_handle.attrs["transverse_axes"])
                monitor_index=int(h5_handle.attrs["index"])
            except Exception:
                h5_handle.close(); raise
        else:
            if isinstance(monitor,(int,np.integer)):
                data=self._monitor_by_id(int(monitor))
            elif isinstance(monitor,dict) and "fields" in monitor:
                data=monitor
            else:
                raise TypeError("monitor must be an ID, loaded monitor dictionary, or HDF5 path.")
            fields=np.asarray(data["fields"]); times=np.asarray(data["time"],dtype=float)
            normal=np.asarray(data["normal"],dtype=float); extent=tuple(data["extent"])
            axes=data["transverse_axes"]; monitor_index=int(data["index"])
        if fields.shape[0] != len(times) or len(times)==0 or fields.shape[-1] != 6:
            if h5_handle is not None: h5_handle.close()
            raise ValueError("Plane monitor has invalid or empty time history.")
        frame_indices=np.arange(0,len(times),stride,dtype=int)
        def frame_values(frame_index):
            return self._plane_time_component(np.asarray(fields[int(frame_index)]),key,normal)
        # Estimate stable global color limits from at most 128 evenly spaced frames.
        scale_positions=np.linspace(0,len(frame_indices)-1,min(128,len(frame_indices)),dtype=int)
        data_min=np.inf; data_max=-np.inf
        for position in scale_positions:
            values=frame_values(frame_indices[position])
            data_min=min(data_min,float(np.nanmin(values)))
            data_max=max(data_max,float(np.nanmax(values)))
        nonnegative=key in {"e","h"}
        if vmin is None and vmax is None:
            if nonnegative:
                vmin=0.0; vmax=max(data_max,1e-30)
            else:
                limit=max(abs(data_min),abs(data_max),1e-30); vmin=-limit; vmax=limit
        elif vmin is None: vmin=data_min
        elif vmax is None: vmax=data_max
        if float(vmax) <= float(vmin):
            if h5_handle is not None: h5_handle.close()
            raise ValueError("vmax must be greater than vmin.")
        if ax is None: _,ax=plt.subplots(figsize=(6,5))
        selected_cmap=cmap or ("viridis" if nonnegative else "RdBu_r")
        first=frame_indices[0]
        image=ax.imshow(frame_values(first).T,origin="lower",extent=extent,aspect="auto",
                        cmap=selected_cmap,vmin=vmin,vmax=vmax)
        ax.set_xlabel(f"{axes[0]} (m)"); ax.set_ylabel(f"{axes[1]} (m)")
        ax.figure.colorbar(image,ax=ax,label=component)
        def update(frame_index):
            image.set_data(frame_values(frame_index).T)
            ax.set_title(f"Monitor {monitor_index}: {component}, t={times[frame_index]:.6g} s")
            return (image,)
        update(first)
        animation=FuncAnimation(ax.figure,update,frames=frame_indices,
                                interval=float(interval),repeat=bool(repeat),blit=False)
        if h5_handle is not None:
            animation._fdtd_h5_handle=h5_handle
            def close_h5(_event):
                if h5_handle.id.valid: h5_handle.close()
            animation._fdtd_close_event=ax.figure.canvas.mpl_connect("close_event",close_h5)
        if save_path is not None:
            destination=Path(save_path).expanduser(); destination.parent.mkdir(parents=True,exist_ok=True)
            save_fps=float(fps) if fps is not None else 1000.0/float(interval)
            writer="pillow" if destination.suffix.lower()==".gif" else None
            kwargs={"fps":save_fps,"dpi":int(dpi)}
            if writer is not None: kwargs["writer"]=writer
            animation.save(destination,**kwargs)
        ax.figure.tight_layout()
        return ax.figure,animation

    view_plane_monitor_animation = animate_plane_monitor

    def plot_slice(self, component="Ez", axis="z", index=None, cmap="RdBu_r", ax=None):
        """Plot a slice of one staggered field component."""
        import matplotlib.pyplot as plt
        if component not in {"Ex","Ey","Ez","Hx","Hy","Hz"}: raise ValueError("Unknown field component.")
        if axis not in "xyz": raise ValueError("axis must be x, y, or z.")
        field=getattr(self,component); a="xyz".index(axis)
        if index is None: index=field.shape[a]//2
        data=np.take(field,int(index),axis=a)
        if ax is None: _,ax=plt.subplots()
        image=ax.imshow(data.T,origin="lower",cmap=cmap,aspect="auto")
        ax.figure.colorbar(image,ax=ax,label=component); ax.set_title(f"{component}: {axis} index {index}")
        ax.figure.tight_layout(); return ax.figure,ax
