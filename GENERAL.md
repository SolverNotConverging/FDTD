# General FDTD guide

This document describes conventions shared by the active 1D, 2D, and 3D
solvers. Use the solver-specific RST documents linked from the root README for
complete method examples.

## Coordinate system and units

- Geometry and grid lengths are in metres.
- Frequencies are in hertz and time values are in seconds.
- `epsilon_r` and `mu_r` are relative, dimensionless constitutive values.
- `sigma_e` is electric conductivity in S/m.
- `sigma_m` is the magnetic-conductivity coefficient used by the symmetric
  lossy Maxwell update.
- Debye `tau` is in seconds. Drude/Lorentz `omega_p`, `omega_0`, and `gamma`
  are angular frequencies in radians per second.
- Integer geometry coordinates generally mean grid indices; floating-point
  coordinates mean physical positions in metres.

All active solvers use Cartesian Yee staggering. Electric and magnetic
components therefore do not occupy the same spatial positions.

| Solver | Electric components | Magnetic components |
|---|---|---|
| 1D | `Ey (Nz+1)` | `Hx (Nz)` |
| 2D TMz | `Ez (Nx+1, Ny+1)` | `Hx (Nx+1, Ny)`, `Hy (Nx, Ny+1)` |
| 2D TEz | `Ex (Nx, Ny+1)`, `Ey (Nx+1, Ny)` | `Hz (Nx, Ny)` |
| 3D | `Ex (Nx, Ny+1, Nz+1)`, `Ey (Nx+1, Ny, Nz+1)`, `Ez (Nx+1, Ny+1, Nz)` | `Hx (Nx+1, Ny, Nz)`, `Hy (Nx, Ny+1, Nz)`, `Hz (Nx, Ny, Nz+1)` |

## Installation and dependencies

The source tree can be imported directly from the project root. Core runtime
dependencies are NumPy, Matplotlib, SciPy, and tqdm. Additional features use:

- Cython and a C compiler for optional compiled kernels;
- Numba plus CUDA for the optional device-resident 2D and 3D GPU backends;
- h5py for 3D plane-monitor persistence;
- Pillow or FFmpeg for exported animations.

Build all optional Cython extensions with:

```bash
python setup_cython.py build_ext --inplace
```

If an accelerated backend is unavailable, the solver falls back to its
Python/NumPy implementation.

## Choosing the grid and time step

Each solver provides `suggest_dx_dt(...)` for an accuracy-oriented starting
point. The shortest wavelength in the model is approximately

\[
\lambda_{\min}=\frac{c_0}{f_{\max}\sqrt{\epsilon_{r,\max}\mu_{r,\max}}}.
\]

Use enough cells per shortest wavelength for the required phase accuracy.
Values around 20–30 are a useful starting range, but resonators, high-index
interfaces, curved objects, and far-field phase calculations may require more.

If `dt` is omitted, each solver selects a value limited by its dimensional CFL
condition and frequency sampling. A user-supplied `dt` must still satisfy the
CFL limit.

## Material-first geometry

All active solvers share the same `Material` definition:

```python
lossy = sim.add_material(
    "lossy",
    epsilon_r=(2.0, 2.5, 3.0),
    mu_r=(1.0, 1.0, 1.0),
    sigma_e=(0.01, 0.02, 0.03),
    sigma_m=0.0,
)
```

The four constitutive arguments can be isotropic scalars or diagonal
`(x, y, z)` sequences. Geometry accepts either the returned `Material` object
or its case-insensitive registered name. `vacuum`, `PEC`, and `PMC` are
registered automatically.

The reduced-dimensional solvers consume only the tensor entries needed by
their polarization:

| Solver | Permittivity/conductivity entries | Permeability/magnetic-loss entries |
|---|---|---|
| 1D Ey/Hx | y | x |
| 2D TMz | z | x and y |
| 2D TEz | x and y | z |
| 3D | x, y, and z | x, y, and z |

Materials are rasterized on cells and averaged to the physical Yee locations.
Curved or partially filled geometry uses subpixel sampling. Direct `ER`, `MR`,
`sigma_e`, and `sigma_m` geometry arguments remain available in 1D/2D for
compatibility, but named definitions are preferred.

### Debye, Drude, and Lorentz dispersion

`epsilon_r` is the instantaneous/high-frequency relative permittivity
`epsilon_inf` whenever dispersive poles are present. A material may contain any
number and combination of the three supported electric pole families:

```python
material = sim.add_material(
    "multipole",
    epsilon_r=(2.0, 2.2, 2.4),
    sigma_e=1e-3,
    debye=[
        {"delta_epsilon": (1.0, 1.1, 1.2), "tau": 10e-12},
        {"delta_epsilon": 0.3, "tau": 40e-12},
    ],
    drude={"omega_p": 2.0e12, "gamma": 8.0e10},
    lorentz={
        "delta_epsilon": 0.75,
        "omega_0": 3.5e12,
        "gamma": 5.0e10,
    },
)
```

Mappings, `DebyePole`, `DrudePole`, and `LorentzPole` objects, or flat short
positional tuples for one isotropic pole are accepted. A list of mappings,
objects, or positional tuples defines multiple poles. Use a mapping or pole
object for one diagonal-anisotropic pole; its parameter values may be Cartesian
triples. Passive materials require
non-negative strengths and damping, positive Debye relaxation time, and
positive Lorentz resonance frequency. PEC and PMC cannot carry poles.
`material.relative_permittivity(omega)` evaluates the diagonal complex model
for angular frequency `omega` using the `exp(-i omega t)` convention;
`sigma_e` remains a separate conductivity term.

With normalized polarization `q=P/epsilon_0`, the continuous auxiliary
equations are

\[
\tau\dot q+q=\Delta\epsilon E,\qquad
\ddot q+\gamma\dot q=\omega_p^2E,\qquad
\ddot q+\gamma\dot q+\omega_0^2q=
\Delta\epsilon\,\omega_0^2E.
\]

The field update solves all poles and ordinary electric conductivity together.
Debye uses a centered trapezoidal recurrence; Drude and Lorentz use the
equivalent trapezoidal first-order polarization/velocity system. This bilinear
discretization is passive/A-stable for passive pole parameters. Each pole owns
its ADE history at the appropriate electric Yee locations. PEC masks clear
both the electric field and its polarization memory.

For partially filled cells, only susceptibility or oscillator forcing strength
is area/volume averaged. Relaxation, collision, and resonance frequencies stay
in separate dynamics channels, so overlapping materials with different poles
are not collapsed into an unphysical averaged frequency.

The ADE constitutive solve always uses the Python/NumPy path. In 1D, a built
Cython kernel can still update H while ADE updates E. The 2D solvers can still
use their Cython curl kernels, but a requested resident-CUDA run falls back to
host updates with a warning. In 3D, a dispersive run requested through either
the Cython or CUDA backend falls back to the NumPy time loop with a warning.
Nondispersive accelerated behavior is unchanged.

Soft electric sources are coupled to the same ADE endpoint update, including
sources placed inside a dispersive object. The 1D matched-source impedance and
the 2D built-in modal eigenproblems currently use `epsilon_inf`; they do not
solve frequency-dependent launch modes. Keep those launch planes in a
nondispersive material, or provide externally calculated modal data where the
2D source API permits it.

## Lossy update coefficients

Electric and magnetic material loss use a centered trapezoidal discretization.
For an electric component,

\[
r_e=\frac{\sigma_e\Delta t}{2\epsilon_0\epsilon_r},\qquad
C_a^E=\frac{1-r_e}{1+r_e},\qquad
C_b^E=\frac{c_0\Delta t}{\epsilon_r(1+r_e)}.
\]

The magnetic coefficients use the corresponding substitution
`sigma_e, epsilon_0, epsilon_r -> sigma_m, mu_0, mu_r`. Setting both
conductivities to zero recovers the original lossless Yee update.

Material conductivity is independent of PML conductivity; one models physical
loss and the other truncates the computational domain.

## PEC and PMC

Perfect conductors are imposed with exact component masks rather than extreme
constitutive values:

- PEC constrains tangential electric components;
- PMC constrains tangential magnetic components.

Use the predefined names directly in geometry, for example
`material="PEC"` or `material="PMC"`.

## Absorbing boundaries

The 2D and 3D solvers use unsplit complex-frequency-shifted convolutional PML
(CFS-CPML). Their default polynomial order is 3, `kappa_max` is 7, and the
target reflection is `R0=1e-8`. The automatically derived peak conductivity is

\[
\sigma_{\max}=-\frac{(m+1)\ln(R_0)}{2\eta_0L}.
\]

The recursive convolution variables modify each affected curl derivative with
the matching `sigma`, `kappa`, and `alpha` profile. Profiles are applied on both
ends of every selected direction and overlap naturally at edges and corners.

The 1D solver currently uses first-order absorbing end conditions rather than
CPML. It also supports exact PEC and PMC end conditions.

Keep sources and measurement surfaces out of the PML. NF2FF surfaces should be
closed, lie in homogeneous background material, enclose every scatterer, and
remain several cells inside the PML interface.

## Backends

- 1D: compiled Cython field updates when available, otherwise Python loops.
- 2D: `config("cpu")` selects Cython curl kernels when built;
  `config("gpu")` selects the persistent Numba-CUDA runtime when available;
  `config("python")` forces the reference implementation. The GPU runtime
  uploads fields, coefficients, CPML state, masks, and sparse source metadata
  once before the run. Curl, CPML, lossy updates, conductor enforcement,
  sources, monitor sampling, and optional history recording then remain on the
  device for the complete time loop. Final fields and requested output buffers
  are copied back after the run, with no host-device transfers per time step.
- 3D: `config("cpu")` selects the compiled whole-run Cython loop when built;
  `config("gpu")` selects a persistent Numba-CUDA loop, and `config("python")`
  uses the NumPy reference implementation. The GPU path uploads the six Yee
  fields, twelve CPML auxiliaries, material coefficients, packed soft-source
  data, and monitor coordinates once per run. It performs no host/device array
  transfers during time stepping, then copies the final mutable state and the
  packed monitor history back once.

Rebuild compiled extensions after changing a `.pyx` file, changing Python or
NumPy versions, or moving to another platform.

For large 2D GPU simulations, prefer line monitors with
`is_include_history=False`. Full-field histories are intentionally accumulated
on the GPU and copied back once. In 3D, only requested plane-monitor samples
are accumulated; they must still fit in device memory. A larger
`record_stride` reduces either allocation.

## Common workflow

1. Estimate resolution and `dt` with `suggest_dx_dt`.
2. Construct the solver and select a backend.
3. Configure absorbing or periodic boundaries.
4. Define materials.
5. Construct geometry using those definitions.
6. Add sources and monitors.
7. Run the simulation.
8. Inspect fields, calculate power, and perform NF2FF where applicable.
9. Save monitor data needed for later analysis.

## Verification

Run the repository tests from the project root:

```bash
python -m unittest discover -s tests -v
```

Every test module is collected under the top-level `tests/` package. The suite
checks Yee shapes and averaging, lossy decay, PEC/PMC masks,
accelerated-backend equivalence, CPML behavior, monitor persistence, power, and
NF2FF output structure.

## Literature supplied with the project

- `doc/Lecture-CPML.pdf`
- `doc/Lecture-Scattering-Analysis.pdf`

Copyright for the supplied literature remains with its original authors.
