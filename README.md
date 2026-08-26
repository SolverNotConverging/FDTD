# FDTD

A research-oriented FDTD project with two deliberately separate solver
families: general-purpose Cartesian electromagnetics and dedicated
curved-spacetime light propagation. Both families have optional Cython and
Numba-CUDA acceleration with portable NumPy fallbacks, but their physical
models, coordinates, units, and APIs are different.

## Choose a solver

### Conventional Cartesian material FDTD

These solvers model user-defined materials, geometry, sources, boundaries, and
monitors in one, two, or three spatial dimensions. They use SI units and
Cartesian Yee grids, with support for anisotropy, electric and magnetic loss,
Debye/Drude/Lorentz dispersion, PEC/PMC geometry, CFS-CPML, power spectra, and
near-field-to-far-field transforms.

| Solver | Polarization and fields | Documentation |
|---|---|---|
| 1D | `Ey`, `Hx` | [FDTD_1D/README.rst](FDTD_1D/README.rst) |
| 2D TMz | `Ez`, `Hx`, `Hy` | [FDTD_2D_Ez/README.rst](FDTD_2D_Ez/README.rst) |
| 2D TEz | `Hz`, `Ex`, `Ey` | [FDTD_2D_Hz/README.rst](FDTD_2D_Hz/README.rst) |
| 3D | `Ex`, `Ey`, `Ez`, `Hx`, `Hy`, `Hz` | [FDTD_3D/README.rst](FDTD_3D/README.rst) |

### Schwarzschild curved-spacetime light solver

`FDTD_2D_GR` is a purpose-built polar TE solver for light on the equatorial
plane of a fixed, non-rotating Schwarzschild black hole. It uses geometric
units (`G=c=1`), evolves `Er`, `Ephi`, and `Hz` through the prescribed GR
optical medium, and has its own orbiting-packet, radial-sponge, diagnostics,
and animation API. It does not use the Cartesian material/geometry/source
workflow above. See [FDTD_2D_GR/README.rst](FDTD_2D_GR/README.rst) for its
physical scope and limitations.

## Quick starts

### Conventional Cartesian example

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

The Cartesian solvers follow the same material-first pattern:

```python
material = sim.add_material(
    "lossy_dielectric",
    epsilon_r=(2.5, 2.5, 3.0),
    mu_r=1.0,
    sigma_e=0.02,
    sigma_m=0.0,
)
```

Here `epsilon_r` is the instantaneous (high-frequency) permittivity. Optional
dispersive poles may be combined and repeated:

```python
dispersive = sim.add_material(
    "dispersive",
    epsilon_r=2.0,
    debye={"delta_epsilon": 1.5, "tau": 8e-12},
    drude={"omega_p": 2.0e12, "gamma": 8.0e10},
    lorentz=[
        {"delta_epsilon": 0.8, "omega_0": 3.0e12, "gamma": 6.0e10},
        {"delta_epsilon": 0.2, "omega_0": 5.0e12, "gamma": 9.0e10},
    ],
)
```

`omega_p`, `omega_0`, and `gamma` are angular frequencies in radians per
second; `tau` is in seconds. Pole parameters can also be Cartesian triples.

`vacuum`, `PEC`, and `PMC` are predefined.

### Schwarzschild GR example

```python
from FDTD_2D_GR import FDTD_2D_GR

sim = FDTD_2D_GR(
    rho_min=0.55, rho_max=10.0, Nr=320, Nphi=640,
).config("cpu")
sim.initialize_orbiting_packet(
    azimuthal_mode=20, radial_width=0.35, angular_width=0.32,
)
history = sim.run(
    duration=0.5 * sim.photon_orbit_period,
    record_stride=8,
)
sim.plot_snapshot(log_scale=True)
sim.plot_diagnostics(history)
```

This is fixed-background Schwarzschild wave propagation rather than a material
scattering model. The finite packet follows the unstable photon-orbit region
before separating into captured and escaping components.

## Build optional Cython kernels

From the project root:

```bash
python setup_cython.py build_ext --inplace
```

The build requires Cython, NumPy, setuptools, and a supported C compiler.
The Cartesian 2D/3D solvers and the Schwarzschild solver have Numba-CUDA
backends that additionally require a working CUDA runtime. Select one with
`sim.config("gpu")`; if CUDA is unavailable, the solver warns and falls back
to its Python/NumPy implementation. Backend details differ by solver; notably,
the current GR Cython kernel is specialized for complex128 fields.

## Run tests

```bash
python -m unittest discover -s tests -v
```

All unit and CUDA-simulator tests live under [`tests/`](tests/). The explicit
start directory also works when the test package is invoked from tools that do
not recursively discover packages by default.

## Examples

### Conventional Cartesian examples

```bash
python FDTD_1D/FDTD_1D_example.py
python FDTD_2D_Ez/Example_1_Simple_Source.py
python FDTD_2D_Hz/Example_1_Simple_Source.py
python FDTD_3D/Example_3D.py
python FDTD_3D/Example_3D_GPU.py
```

Both 3D examples default to a `100 × 100 × 100` grid. They share the same
scattering model and post-processing so CPU and GPU output can be compared
directly. Use `--steps`, `--record-stride`, `--output-dir`, `--animate`, and
`--no-show` to control a run; `--cells` may increase, but not reduce, the grid.

The `FDTD_2D_Ez_Legacy` directory is retained for reference. New work should
use `FDTD_2D_Ez` or `FDTD_2D_Hz`.

### Schwarzschild GR example

```bash
python FDTD_2D_GR/Example_Photon_Orbit.py
```

This is a no-argument Python API script. Edit its constants directly;
`BACKEND` selects NumPy, Cython, or CUDA, and `SAVE_ANIMATION = True` writes
`photon_packet.mp4` with FFmpeg after the simulation finishes.

## Further documentation

Start with [GENERAL.md](GENERAL.md) for installation, units, stability,
backend selection, and conventions shared where applicable. The supplied CPML
and scattering-analysis literature is indexed in [doc/README.md](doc/README.md).

## License

See [LICENSE](LICENSE).
