# FDTD
A compact, research-oriented FDTD (finite-difference time-domain) solver for Maxwell's equations. It supports 1D, 2D, and 3D time-domain simulations with CPML absorbing boundaries, anisotropic materials, multiple source types, and frequency-domain post-processing. GPU acceleration is available for the unified 2D solvers through Numba-CUDA, while the complete 3D update/source/monitor loop has an optional Cython backend.

**What It Does**
- Solves 1D, 2D, and 3D electromagnetic wave propagation in the time domain.
- 2D modes include TMz (`FDTD_2D_Ez`) and TEz (`FDTD_2D_Hz`).
- CPML boundaries for low-reflection truncation.
- Named material modeling with anisotropy, electric/magnetic loss, and subpixel-smoothed geometry.
- Native PEC/PMC geometry masks with Yee-component update constraints.
- Source options: point, line-soft, TF/SF, and waveguide eigenmode ports.
- Monitors and analysis: line monitors, FFT-based power, and 2D NF2FF.
- Unified 2D CPU/GPU solvers selected with `sim.config(backend=...)`.
- Optional Cython CPU kernels and Numba-CUDA GPU curl kernels, both with Python fallback.

**Project Layout**
- `FDTD_1D`: 1D solver and example (`FDTD_1D_example.py`).
- `FDTD_2D_Ez`: unified CPU/GPU 2D TMz solver and examples.
- `FDTD_2D_Hz`: unified CPU/GPU 2D TEz solver and examples.
- `FDTD_2D_Ez_Legacy`: older 2D Ez implementation and examples.
- `FDTD_3D`: full-vector 3D Yee solver with geometry, CPML, plane monitors, power, and NF2FF.
- Cython kernel sources:
  - `FDTD_1D/cython_kernel_1d.pyx`
  - `FDTD_2D_Ez/cython_kernel_ez.pyx`
  - `FDTD_2D_Hz/cython_kernel_hz.pyx`
  - `FDTD_3D/cython_kernel_3d.pyx`

**Quick Start**
```bash
python FDTD_1D/FDTD_1D_example.py
python FDTD_2D_Ez/Example_1_Simple_Source.py
python FDTD_2D_Hz/Example_1_Simple_Source.py
```

## CPU and GPU Backends

### 1D Cython kernel

`FDTD_1D` uses a Yee-staggered grid with `Nz + 1` electric samples and `Nz`
magnetic samples. It imports the compiled Cython update kernel when one is
present and automatically falls back to equivalent Python loops otherwise.
Its objects and boundaries accept native PEC/PMC constraints without replacing
ER or MR with artificial large values:

```python
sim.add_material("lossy_glass", epsilon_r=4.0, sigma_e=0.02)
sim.add_object(material="lossy_glass", region=(2e-3, 4e-3))
sim.add_object(material="PEC", region=(4e-3, 5e-3))
sim.add_object(material="PMC", region=slice(200, 210))
sim.set_boundary(left="PEC", right="PMC")
```

The TMz and TEz solvers store material on `(Nx, Ny)` cells, rasterize geometry
with 16×16 subpixel sampling by default, and then average each tensor component
onto its exact Yee face or center. Their 1D FDFD port eigensolvers use matching
cell-to-face material averaging and rectangular staggered difference operators.
Waveguide modes eliminate PEC/PMC cell unknowns with a reduced-DOF
eigenproblem and impose the corresponding Dirichlet or Neumann boundary.
TMz uses nodal `Ez (Nx+1, Ny+1)`, `Hx (Nx+1, Ny)`, and `Hy (Nx, Ny+1)`;
TEz uses `Hz (Nx, Ny)`, `Ex (Nx, Ny+1)`, and `Ey (Nx+1, Ny)`.

Build all optional extensions from the repository root:

```bash
python setup_cython.py build_ext --inplace
```

This build requires Cython, NumPy, setuptools, and a platform C compiler. The
2D solvers select a backend through the same public class:

```python
from FDTD_2D_Ez import FDTD_2D_Ez

sim = FDTD_2D_Ez(x_range=1e-3, y_range=1e-3, Nx=50, Ny=50, f_max=1e11, Nt=10)
sim.config(backend="cpu")  # Cython when built, otherwise Python loops
print(sim.backend)

sim.config(backend="gpu")  # Numba-CUDA when available, otherwise Python loops
print(sim.backend)
```

The same API works for `FDTD_2D_Hz`. Numba-CUDA is optional and PyTorch is not
required.

### Named Materials, Loss, and 2D Geometry

Define materials before constructing geometry. `epsilon_r`, `mu_r`, `sigma_e`,
and `sigma_m` may be scalars or diagonal `(x, y, z)` values. All 2D shape
functions first rasterize them onto material cells, use the solver's subpixel
setting (16 samples per axis by default), and then average each property onto
its exact Yee location:

```python
sim.add_material("lossy_slab", epsilon_r=4.0, mu_r=1.0,
                 sigma_e=0.02, sigma_m=0.0)
sim.add_material("lens", epsilon_r=2.5)
sim.add_material("prism", epsilon_r=(3.0, 3.2, 3.5))

sim.add_rectangle(material="lossy_slab",
                  x_position=(0.2e-3, 0.4e-3),
                  y_position=(0.2e-3, 0.8e-3))
sim.add_circle(material="lens", center=(0.7e-3, 0.5e-3), radius=0.1e-3)
sim.add_triangle(material="prism",
                 vertices=((0.1e-3, 0.1e-3),
                           (0.4e-3, 0.1e-3),
                           (0.2e-3, 0.35e-3)))
```

TMz consumes the z electric and x/y magnetic tensor entries; TEz consumes
the x/y electric and z magnetic entries. The Yee updates use centered
(trapezoidal) conductivity terms, including both electric and magnetic field
decay. Direct `ER=`, `MR=`, `sigma_e=`, and `sigma_m=` shape arguments remain
available for older scripts.

Pass `material="PEC"` or `material="PMC"` to any of these functions for a
perfect conductor. Conductor cells are selected by their centers, `subpixel`
is ignored, ER/MR stay unchanged, and the surrounding Yee update coefficients
are constrained directly:

```python
sim.add_rectangle(material="PEC", x_position=(0.0, 0.05e-3),
                  y_position=(0.1e-3, 0.9e-3))
sim.add_triangle(material="PMC",
                 vertices=((0.6e-3, 0.2e-3),
                           (0.9e-3, 0.2e-3),
                           (0.75e-3, 0.45e-3)))
```

Animations outline every PEC region with a dashed yellow boundary and every
PMC region with a dashed blue boundary in both the 1D and 2D solvers.

### Monitor IDs, Partial NF2FF Surfaces, and Plane Power

Each 2D line monitor may be given a stable non-negative ID. NF2FF and power
functions resolve this ID rather than its position in the monitor list:

```python
sim.add_line_monitor(x=(20, 120), y=120, index=10)  # top
sim.add_line_monitor(x=(20, 120), y=20, index=20)   # bottom
sim.add_line_monitor(y=(20, 120), x=120, index=40)  # right
```

Any NF2FF side may be omitted with `None`, provided at least one monitor is
present. This is useful when a PEC plane removes the fields behind one side:

```python
ff = sim.NF2FF(top=10, bottom=20, left=None, right=40,
               freqs=[80e9, 100e9], src_index=0)
```

`power_spectrum` evaluates a direct DFT at exactly the requested frequencies,
integrates the signed Poynting flux across the full monitor plane, and divides
it by the selected source-power spectrum (including its aperture/modal geometry
factor). `plot_power_spectrum` plots this dimensionless power ratio against
frequency:

```python
plane_power = sim.power_spectrum(
    monitor_index=40,
    freqs=[80e9, 90e9, 100e9],
    source_index=0,
)
sim.plot_power_spectrum(plane_power)
```

## 3D Yee Solver

The 3D solver stores every component at its physical Yee location:

- `Ex (Nx, Ny+1, Nz+1)`, `Ey (Nx+1, Ny, Nz+1)`, `Ez (Nx+1, Ny+1, Nz)`
- `Hx (Nx+1, Ny, Nz)`, `Hy (Nx, Ny+1, Nz)`, `Hz (Nx, Ny, Nz+1)`

Define named materials first, then use those definitions to construct geometry.
Relative permittivity, relative permeability, and electric/magnetic conductivity
may be isotropic scalars or diagonal `(x, y, z)` values. `PEC` and `PMC` are
built in and are imposed as exact Yee-component masks:

```python
from FDTD_3D import FDTD_3D

sim = FDTD_3D(20e-3, 20e-3, 20e-3, 40, 40, 40,
              f_min=8e9, f_max=12e9, Nt=600)
sim.config("cpu")
sim.add_PML(6)

glass = sim.add_material("glass", epsilon_r=4.0, mu_r=1.0,
                         sigma_e=1e-4)
sim.add_block(glass, x=(4e-3, 8e-3), y=(5e-3, 15e-3), z=(7e-3, 12e-3))
sim.add_cylinder("glass", center=(13e-3, 10e-3, 10e-3),
                 radius=2e-3, height=8e-3, axis="z")
sim.add_sphere("PEC", center=(10e-3, 10e-3, 14e-3), radius=1.5e-3)
```

Geometry spans accept floats in metres or integers as cell-edge indices. Sphere
and cylinder centers use the same convention. Ordinary curved geometry uses
subpixel volume sampling; the constructor's `subpixel` setting defaults to 4
samples per voxel axis.

Point, line, and plane sources are additive soft electric sources. A tuple marks
a half-open span, so the number of spans must be zero, one, or two respectively:

```python
sim.add_source("point", x=20, y=20, z=10, polarization="z")
sim.add_source("line", x=(8, 32), y=20, z=10, polarization="x")
sim.add_source("plane", x=(8, 32), y=(8, 32), z=10,
               polarization="x", amplitude=1.0)
```

Plane monitors interpolate all six staggered components to voxel centers. Their
stable IDs can be used for signed, area-integrated Poynting power. A convenience
function creates the six outward-facing monitors needed by the 3D surface-
equivalence NF2FF transform:

```python
box = sim.add_nf2ff_box(x=(10, 30), y=(10, 30), z=(12, 32), start_index=100)
sim.run(record_stride=1)  # tqdm displays Cython/NumPy time-step progress

power = sim.power_spectrum(box["z_max"], freqs=[8e9, 10e9, 12e9],
                           source_index=0, window="hann")
sim.plot_power_spectrum(power, db=True)

ff = sim.NF2FF(box, freqs=[10e9], source_index=0, window="hann")
sim.plot_nf2ff(ff, db=True, db_floor=-40)  # 3D dB radiation surface
```

`run()` shows a `tqdm` progress bar for both the Cython and NumPy backends.
Pass `progress=False` for batch jobs where progress output is not desired. The
compiled backend is advanced in coarse progress chunks while preserving source
timing and monitor `record_stride` alignment.

Monitor histories can optionally be saved automatically when `run()` finishes.
The chunked, gzip-compressed HDF5 file contains all six fields, sample times,
physical coordinates, surface normal, and plane geometry. It can be reloaded in a later session for
time-domain plots, frequency-domain plots, power, or NF2FF analysis:

```python
monitor_id = sim.add_plane_monitor(
    axis="z", position=30, first=(8, 32), second=(8, 32),
    index=5, normal="+", save_path="output/monitor_data/z30.h5",
)
sim.run()  # output/monitor_data/z30.h5 is written automatically

# Explicit saving is also available:
sim.save_plane_monitor(monitor_id, "output/monitor_data/z30_copy.h5")

# Later, or in a newly constructed solver:
saved = sim.load_plane_monitor("output/monitor_data/z30.h5",
                               index=105, register=True)
sim.plot_plane_monitor(saved, component="Ez", time_index=-1)
sim.plot_plane_monitor(saved, component="Ez", frequency=10e9,
                       representation="magnitude", window="hann")

# Animate in memory or stream frames directly from the HDF5 file:
fig, animation = sim.animate_plane_monitor(
    "output/monitor_data/z30.h5", component="Ez",
    frame_stride=2, interval=40,
)

# Optionally export the animation:
sim.animate_plane_monitor("output/monitor_data/z30.h5", component="Ez",
                          save_path="output/monitor_data/z30.gif", fps=25)
```

`plot_plane_monitor` and `animate_plane_monitor` accept a registered monitor ID,
a loaded monitor dictionary, or the HDF5 filename directly. Components `Ex`,
`Ey`, `Ez`, `Hx`,
`Hy`, `Hz`, vector magnitudes `E`/`H`, and normal Poynting flux `Snormal` are
supported. GIF export uses Pillow; other animation formats use the writers
available to Matplotlib, such as FFmpeg for MP4.

HDF5 monitor persistence requires `h5py`; GIF export additionally requires
Pillow.

The monitor Fourier transform applies the Yee half-time-step phase correction
to `H` before calculating power or equivalent currents. Keep an NF2FF surface
in homogeneous background material, outside all scatterers and inside the PML.
See `FDTD_3D/Example_3D.py` for a complete workflow.

### Examples Policy
- `GPU_exmaple.py` files select `backend="gpu"` on the standard solver classes.
- Other examples explicitly select `backend="cpu"`.

### Cross-Platform Notes
- If Cython or Numba-CUDA is unavailable, the corresponding simulation still runs via Python loops.
- Rebuild compiled kernels after moving the project between platforms or Python environments.

**Typical Workflow**
1. Create a solver with domain size, grid resolution, and frequency bounds.
2. Configure boundaries (CPML/periodic).
3. Add geometry (rectangles, circles, triangles, or custom material regions).
4. Add sources (point, line-soft, TF/SF, or waveguide modes).
5. Add monitors if needed (line monitors, NF2FF, FFT power).
6. Run, visualize, and post-process (animations and FFT plots).


⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠁⡇⠀⠀⠀⠀⠀⠀⠀⠸⣿⡇⠀⠚⠉⣸⣿⣅⣀⠀⠀
⠙⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⢸⠀⠀⠀⠀⠀⠀⠀⠀⢠⣿⣷⣾⠿⠿⠿⠿⠛⠉⠀⠀
⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⡎⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠟⠛⠛⠛⠛⠛⠛⢛⠛⠛⠻⠿⠿⠿⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣦⣀⡀⠀⠀⠀⠀⢠⡿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰
⣿⣿⣿⣿⣿⣿⠿⠛⠛⠉⠉⢀⣤⠞⠋⠀⠀⠀⠀⢀⡤⢀⡤⠚⠁⠀⠀⠀⠀⠀⠀⣠⠴⠂⠀⠈⠉⠉⠛⠻⠿⣿⣿⣿⣿⣿⣿⣿⣶⣤⡀⢀⡟⠁⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⣿
⣿⣿⣿⣿⠏⣠⠀⠀⠀⠀⣰⠟⠁⠀⠀⠀⠀⣠⠖⠉⠀⠀⠀⠀⠀⠀⠀⠀⢀⡴⠚⠁⠀⠀⠀⠀⠀⠀⠀⠀⢀⡟⠈⠉⠛⠿⣿⣿⣿⣿⣿⣿⣄⡀⠀⠀⠀⠀⠀⠀⠀⣠⣾⣿⣿
⣿⣿⣿⢏⡴⠃⠀⠀⢠⡞⠁⠀⣠⠔⠀⣠⢞⣡⠞⠁⠀⠀⠀⠀⠀⠀⢀⡴⠋⠀⠀⠀⠀⠀⠀⠀⣠⠄⠀⡠⢻⠁⠀⠀⠀⠀⠀⠙⠻⢿⣿⣿⣿⣿⣦⡀⠀⠀⠀⢀⣴⣿⣿⠉⠉
⣿⣿⡷⠋⠀⠀⠀⣰⠋⣀⠴⠋⠀⣠⠞⢡⠟⠁⠀⠀⠀⠀⠀⠀⠀⣠⠞⠁⠀⠀⠀⠀⠀⣀⡴⠊⢁⡴⠊⢁⡏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠻⣿⣿⣿⣿⣦⡀⣠⣾⣿⡟⠉⠀⢀
⡿⠏⡀⠀⢀⣠⣾⠷⠋⠁⠀⣠⠞⠁⢠⠏⠀⠀⠀⠀⠀⠀⠀⢀⡼⠃⠀⠀⠀⠀⣀⡴⠚⠁⣠⠔⠋⠀⠀⣸⠁⠀⠀⠀⠀⠀⠀⠀⠀⣀⠀⣸⠁⠙⢿⣿⣿⣿⣿⣿⣿⡀⠀⠀⠃
⣁⣠⠵⠞⠛⠉⠀⠀⠀⣠⡾⣡⠄⢠⡟⠀⠀⠀⠀⠀⠀⠀⢠⡞⠀⠀⢀⣠⠴⠚⢁⣤⣶⣋⡁⠀⠀⠀⢀⡇⠀⠀⠀⠀⠀⠀⠀⠀⡼⠁⢠⡇⠀⠀⠀⠙⢿⣿⣿⣿⣿⣷⣦⡤⠂
⠁⠀⠀⠀⠀⢀⣠⢴⡾⣫⣾⠃⢠⣿⠁⠀⠀⠀⠀⠀⡀⢠⣏⡠⠴⢚⣉⠤⠖⠛⠁⠀⠀⠀⠈⠙⠳⠦⣼⠁⠀⠀⠀⠀⠀⠀⠀⡜⠁⢠⢿⡇⠀⡜⢠⠀⠀⠙⢿⣿⣿⣿⣿⠀⠀
⠀⠀⠀⠀⢠⠞⢁⣞⡽⢡⡏⠀⡞⡞⠀⠀⠀⠀⠀⠀⢷⣿⠵⠒⣯⡉⠀⢠⡄⠤⠤⠤⣀⣀⠀⠀⠀⠀⡾⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⡟⢸⠁⠀⠀⡟⢠⢀⠀⡀⠻⣿⣿⣿⣷⣾
⠀⠀⠀⠀⠀⢀⣾⡿⠁⣼⠀⢸⢡⡇⠀⠀⠀⠀⠀⠀⡞⣿⣿⡿⠿⠿⠿⠿⠿⣿⣿⣶⣤⣄⠉⠳⣄⡀⡇⠀⠀⠀⠀⠀⠀⠀⡼⣲⠋⡇⢸⠀⠀⢰⠃⡆⢸⢠⢹⠀⠙⣿⣿⣿⡏
⠀⠀⢀⣠⠴⠛⡿⣧⣴⡏⠀⡏⢸⠇⠀⠀⠀⠀⠀⢸⠁⢹⡇⠀⠀⣠⠖⢻⣟⠲⢮⡙⢿⣿⡗⢆⠘⡅⡇⠀⠀⠀⠀⠀⢀⡞⣰⠇⠀⣷⢸⠀⠀⣾⠀⢧⠈⣿⢸⢠⠀⠸⣿⣿⣿
⣶⣚⣉⠤⠤⠾⣧⠘⣇⣷⠀⠃⢸⡀⠀⠀⠀⠀⠀⣿⠀⠘⡇⠀⡾⠥⠞⠉⠁⠹⣾⣷⠀⠘⢿⡌⠀⢹⡇⠀⠀⠀⠀⠀⡼⢠⠏⠉⠛⢿⣼⠀⢰⣇⠀⠘⠦⣿⠀⢿⠀⠀⢿⣿⣿
⠀⠀⠀⠀⠀⢀⢿⣦⡈⠻⣇⠀⠸⡇⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⢸⣼⠀⠀⠈⠧⠀⠀⡇⠀⠀⠀⠀⢸⢣⡏⠀⠀⠀⢸⣿⠀⢸⢻⠀⠀⢸⣿⡀⠈⠐⡆⢸⣿⣿
⠀⠀⠀⠀⠀⠁⢈⣯⢳⣄⠘⡄⠀⢷⠀⠀⠀⢸⡀⢹⡀⠀⠀⠀⠹⠤⣀⡀⠀⠀⢀⡟⠀⠀⠀⠀⠀⠀⣷⠀⠀⢰⠂⡏⡾⠀⠀⣀⠀⠀⣿⡇⡾⣾⡄⠀⢸⠃⠃⠀⠀⣧⢸⣿⣿
⠀⠀⠀⠀⠀⣠⠞⢈⣿⡉⠛⣿⣆⠘⣆⠀⠀⠀⢳⡀⢷⡀⠀⠀⠀⠀⠀⠉⠓⠚⠉⠀⠀⠀⠀⠀⠀⠀⢸⡄⠀⢸⣶⣿⠇⣀⡬⢭⣉⠲⣼⣇⡇⠘⡇⢐⡿⠀⠀⠀⢰⡟⢸⣿⣿
⢦⣀⠀⣠⣾⠏⢀⣾⣿⡇⠀⣿⢯⣧⣘⢦⡀⠀⠀⢻⣦⣳⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣧⠀⢸⢻⣿⣴⣾⣿⡿⢿⣷⣌⢿⡇⠀⣧⣼⠁⠀⠀⠀⣼⡇⣸⣿⣿
⡄⠈⠛⠳⠧⣤⣾⠿⠋⠀⠀⡿⠈⢣⡘⣿⠿⠦⣤⣬⡄⠈⠙⠛⠛⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⡆⢿⠈⣿⡯⠿⠳⣝⣦⠙⣿⡾⠃⠀⣿⠃⠀⠀⠀⢠⡿⢡⣿⣿⣿
⠛⠷⣤⡀⠀⠀⠀⠀⠀⠀⣰⠇⡇⠀⢳⡘⢦⠀⠀⠀⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢳⣸⡾⠁⠱⠄⠀⢻⣿⠀⠸⣿⣖⡾⠃⠀⠀⠀⠀⡼⢣⣾⣿⣿⣿
⠀⠀⠈⠛⢶⣤⣤⣤⠤⢾⣿⡄⢡⠀⠀⠙⣎⢳⡀⠀⢸⡄⠀⠀⠀⠀⢀⣀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣿⣄⠀⢀⡐⢻⡟⠀⢠⣿⡿⠷⠄⠀⢀⣠⢞⣵⠿⣿⣟⢻⣿
⣦⡀⠀⠀⠀⢻⣿⣦⣴⠋⡏⣇⠘⢧⡀⠀⣏⣷⣽⣦⣄⠹⣄⠀⠀⢀⡟⠃⠈⠉⠹⠗⠢⣄⡀⠀⠀⠀⠀⠀⠀⠀⠙⠻⠷⢤⠤⠊⢀⣴⡿⠋⢀⡼⠒⠚⠿⠚⢋⡇⠘⣿⠻⣿⣿
⢹⣿⣦⡀⠀⠀⠹⣿⣿⠀⠹⣼⡄⠈⠛⡄⢹⣼⡈⢿⠈⠙⠛⠓⠒⠘⡇⠀⠀⠀⠀⠀⠈⠲⡍⠳⢄⡀⠀⠀⠀⠀⠀⠀⠀⠠⢄⣤⣖⣋⣤⢶⡿⠁⠀⠀⢀⡾⣸⠀⠀⣏⠀⠙⣟
⠀⠈⢿⡷⣄⠀⠀⢿⣿⣧⠀⠹⣧⠀⠀⠃⠆⢿⣧⠈⠀⠀⠀⠀⠀⠀⢣⠀⠀⠀⠀⠀⠀⠀⠈⠆⠀⠹⡄⠀⠀⠀⠀⠀⠀⠀⠀⣰⢿⣟⡵⠋⠀⠀⠀⢀⣾⠁⡟⠀⠀⣿⡀⠀⢈
⠀⢀⣽⣧⠈⢷⡀⠸⡎⢻⣷⡀⠘⢧⡀⠀⠀⠸⣿⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡼⠁⠀⠀⠀⢠⣀⠀⢀⣾⣿⠿⠋⠀⠀⠀⠀⣠⣟⣽⣸⠁⠀⠀⣿⡇⠀⠀
⡁⠊⢿⣿⠀⠀⠙⡄⢷⠀⠙⣿⣦⡈⠛⣦⡀⠀⠹⣿⢧⡀⠀⠀⠀⠀⠀⠀⠀⠀⠠⣄⣀⣠⠤⠚⠉⠀⠀⠀⠀⠀⠀⠙⣿⣭⡉⠀⠀⠀⢀⣠⣴⣿⣿⣿⣿⡏⠀⠀⢸⣿⡇⠀⠐
⣶⣿⣿⣿⡆⠀⠀⡀⢸⡆⠀⠈⢻⣿⣦⡀⠙⢦⣀⢻⡀⠹⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⢾⠁⢸⢹⠙⠛⢉⠹⣿⣿⣿⣿⣿⣿⣆⠀⠀⣾⡇⢹⠀⠀
⣿⣿⣿⣿⡇⠀⠀⣷⠀⡇⠀⢀⡀⢿⣿⣿⣦⡀⠙⠻⣿⣆⠈⢷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⣤⠴⠚⠋⠀⢸⠀⣿⠘⡆⠀⢠⢸⣿⢿⣿⣿⣿⠿⣿⣆⣾⡟⠀⢸⠀⡇
⣿⣿⣿⣿⡇⠀⠀⡟⠀⣷⠀⡾⠀⠘⣿⣿⣿⣷⡄⠀⠈⠻⣧⡀⣹⡷⢤⣀⣤⡤⠶⠶⠞⠛⠋⠉⠉⠁⢀⠀⠀⠀⠀⢸⠀⣿⠀⢱⡀⣘⢸⢘⣿⣿⣿⡋⠄⠿⢿⣯⠀⠀⣼⡐⢳
⣿⣿⣿⣿⡇⠀⢀⡇⠀⢺⠀⡇⠀⠀⢻⣿⣿⣿⣿⡀⠀⠀⠈⠻⣿⣷⣾⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡀⠀⠀⠀⢸⠀⡇⠀⠀⢳⡈⢹⡉⠸⡽⣜⣧⠀⠀⠀⠙⢷⡀⠻⠒⠋
⣿⣿⣿⣿⣧⠀⣸⠁⠀⢸⢸⠃⠀⠀⢸⣿⣿⣿⣿⣷⠀⠀⠀⢀⡘⣿⣿⡿⠳⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠃⠀⠀⠀⠈⡇⣧⠀⠀⠀⠹⣌⢧⠀⢳⠙⣎⢷⡀⠀⠀⠈⠻⣄⠀⠀
⣿⣿⣿⣿⡿⢀⡏⠀⠀⣼⣾⠀⠀⠀⠈⣿⣿⣿⣿⣿⣇⠀⠀⠀⠙⠞⢞⢿⣖⣿⣷⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢳⢻⡀⠀⠀⠀⠈⠻⣷⣤⣧⠘⢯⣳⣄⠀⠀⠀⠙⢧⡀
