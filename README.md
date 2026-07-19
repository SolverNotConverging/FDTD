# FDTD

A research-oriented Python implementation of one-, two-, and three-dimensional
finite-difference time-domain solvers on Yee-staggered grids.

The project supports named anisotropic materials, electric and magnetic loss,
native PEC/PMC geometry, CFS-CPML boundaries in 2D and 3D, multiple source
types, field monitors, power spectra, and near-field-to-far-field transforms.
Optional Cython and Numba-CUDA backends accelerate the main numerical work
while retaining portable NumPy/Python fallbacks.

## Documentation

Start with [GENERAL.md](GENERAL.md) for installation, units, material
conventions, stability guidance, backend selection, and the common workflow.
Each active solver has its own detailed reference:

| Solver | Polarization and fields | Documentation |
|---|---|---|
| 1D | `Ey`, `Hx` | [FDTD_1D/README.rst](FDTD_1D/README.rst) |
| 2D TMz | `Ez`, `Hx`, `Hy` | [FDTD_2D_Ez/README.rst](FDTD_2D_Ez/README.rst) |
| 2D TEz | `Hz`, `Ex`, `Ey` | [FDTD_2D_Hz/README.rst](FDTD_2D_Hz/README.rst) |
| 3D | `Ex`, `Ey`, `Ez`, `Hx`, `Hy`, `Hz` | [FDTD_3D/README.rst](FDTD_3D/README.rst) |

The supplied CPML and scattering-analysis literature is indexed in
[doc/README.md](doc/README.md).

## Quick start

```python
from FDTD_3D import FDTD_3D

sim = FDTD_3D(
    x_range=20e-3, y_range=20e-3, z_range=20e-3,
    Nx=40, Ny=40, Nz=40, f_min=8e9, f_max=12e9, Nt=600,
)
sim.config("cpu")
sim.add_PML(6)

glass = sim.add_material("glass", epsilon_r=4.0, sigma_e=1e-4)
sim.add_block(glass, x=(5e-3, 9e-3), y=(5e-3, 15e-3), z=(8e-3, 12e-3))
sim.add_sphere("PEC", center=(13e-3, 10e-3, 10e-3), radius=1.5e-3)

sim.add_source("plane", x=(10, 30), y=(10, 30), z=8, polarization="x")
monitor = sim.add_plane_monitor("z", position=30, index=1)
sim.run(record_stride=2)
sim.plot_plane_monitor(monitor, component="Ex", time_index=-1)
```

All solvers follow the same material-first pattern:

```python
material = sim.add_material(
    "lossy_dielectric",
    epsilon_r=(2.5, 2.5, 3.0),
    mu_r=1.0,
    sigma_e=0.02,
    sigma_m=0.0,
)
```

`vacuum`, `PEC`, and `PMC` are predefined.

## Build optional Cython kernels

From the project root:

```bash
python setup_cython.py build_ext --inplace
```

The build requires Cython, NumPy, setuptools, and a supported C compiler.
The 2D GPU backend additionally requires Numba and a working CUDA runtime.

## Run tests

```bash
python -m unittest discover -v
```

## Examples

```bash
python FDTD_1D/FDTD_1D_example.py
python FDTD_2D_Ez/Example_1_Simple_Source.py
python FDTD_2D_Hz/Example_1_Simple_Source.py
python FDTD_3D/Example_3D.py
```

The `FDTD_2D_Ez_Legacy` directory is retained for reference. New work should
use `FDTD_2D_Ez` or `FDTD_2D_Hz`.

## License

See [LICENSE](LICENSE).
