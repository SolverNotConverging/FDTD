# Geometry-first general 2D reference solver

`FDTD_2D.FDTD2D` combines TE (`Hz`, in-plane E) and TM (`Ez`, in-plane H)
in a shared nonuniform finite-integration runtime. Define the physical scene
and ports first, generate the mesh second, and compile the field topology last.
All lengths are metres, frequencies hertz, and time seconds.

This is a new CPU reference implementation. The existing uniform Yee solvers
and their compiled/CUDA kernels retain their APIs. The new scalar arrays are
cell-centered, with additional scalar unknowns in cells split by sheets; TM
therefore uses the dual layout rather than the legacy vertex-centered `Ez`.
The conformal reference is intended for further numerical validation before
production accuracy claims or accelerator implementation.

## Install and run

```bash
python -m pip install -r FDTD_2D/requirements.txt
python -m FDTD_2D.examples.example_general
python -m FDTD_2D.examples.example_waveguide
python -m FDTD_2D.examples.example_scattering
python -m unittest tests.test_general_2d -v
```

The examples save compressed NumPy S-parameter data. Array ordering is
`s[frequency, outgoing_channel, incoming_channel]`; channel labels are
`(port_name, mode_index)`. Lumped ports use mode index zero. Invalid bins remain
NaN and are identified by `valid`, rather than being replaced by zeros.

`run()` and `scattering()` show terminal progress by default. Pass
`progress=False` to disable it. A callable `progress(info, scalar, vector)`
receives read-only field views alongside the terminal display, including when
the GUI uses the callback to report its own progress.

## Geometry before meshing

```python
import numpy as np
from FDTD_2D import (
    Scene, Material, MeshPolicy, FDTD2D, PEC, PMC,
    SurfaceImpedance, ThinSheet, LumpedPort, GaussianPulse, RunControl,
)

scene = Scene()
scene.rectangle("substrate", (.012, .026), (.008, .022),
                Material("n3", epsilon_r=9), rank=20)
scene.polygon("tip", [(.010, .010), (.014, .015), (.010, .020)], PEC, rank=40)
scene.sheet("film", (.028, .010), (.030, .020),
            ThinSheet.resistive(75), rank=30)
scene.add_port(LumpedPort("feed", (.016, .015), rank=100))
scene.add_port(LumpedPort("receive", (.026, .015), rank=100))

policy = MeshPolicy(max_step=.002, f_min=5e9, f_max=15e9,
                    min_step=.00025, min_dt=2e-13, growth=2)
sim = FDTD2D(scene, mesh_policy=policy, polarization="TM",
             frequencies=np.linspace(5e9, 15e9, 51))
study = sim.scattering(RunControl(max_time=4e-9),
                       GaussianPulse(10e9, 8e-11))
```

Rectangles, tessellated circles, arbitrary valid polygons (including holes
through `scene.add`), and straight zero-thickness sheets are supported.
`rank` affects mesh requests; larger values win. It does not change physical
material overlays, which follow insertion order. A later dielectric can erase
part of an earlier conductor. The compiled solver snapshots the scene, so
subsequent scene edits require a new solver instance.

Omit both domain axes to generate the simulation after placing the objects.
`DomainPolicy` encloses the geometry and port apertures, adds a closed TF/SF
rectangle when plane waves are present, surrounds it with a closed NTFF
rectangle, then adds background clearance and an outer CPML collar. Clearance
uses the background wavelength at `f_min` (or `f_max` when omitted); the policy
also enforces cell-based minimum margins. Inspect `sim.scene.generated` for all
bounds. The caller's `Scene()` remains independent of the generated mesh.
Explicit domains and meshes remain available for controlled numerical studies.

```python
from FDTD_2D import DomainPolicy
sim = FDTD2D(scene, mesh_policy=policy, polarization="TM",
             frequencies=np.linspace(5e9, 15e9, 51),
             domain_policy=DomainPolicy(pml_cells=10, clearance_wavelengths=.3))
```

## Mesh policy and conflicts

The local wavelength target is

\[
  h \le \frac{c_0}{N_\lambda f_{\max}\max_{f,\alpha}\operatorname{Re}
                    \sqrt{\epsilon_{r,\alpha}(f)\mu_{r,\alpha}(f)}}.
\]

The index maximum is sampled across `f_min..f_max`, with extra samples near
Lorentz resonances. If `f_min` is omitted, the sampling band begins at
`f_max/1000`. This samples the material model; it is not a rigorous global
optimization of an arbitrarily narrow dispersive resonance. Although meshing
understands the shared dispersive material definitions, bulk ADE time stepping
is not yet implemented in this runtime.

Requests include geometry bounding boxes, polygon vertices/tips, port
coordinates, narrow extents, and closest-point gaps between conductor objects.
Matched waveguide ports also request a uniform one-cell collar on each side of
the aperture. `mesh_step` optionally selects its longitudinal spacing. If a
higher-ranked conflicting geometry prevents that collar, compilation reports
the unresolved aperture instead of moving it.
High-index regions request finer intervals on both axes. Rectilinear lines
span the domain, so projected refinement also affects cells outside the object.
The policy currently recognizes gaps between distinct objects; detecting a
narrow reentrant gap within one polygon remains an extension.

`min_step` and the conservative vacuum-speed floor derived from `min_dt`
prevent nearly coincident anchors from creating prohibitively small cells.
Within each axis, requests are processed by descending rank, then by position
and name for deterministic ties. Domain endpoints are mandatory. Identical
coordinates share an anchor. Conflicting lower-rank requests are rejected and
recorded with the winning request. Geometry is still compiled at its original
physical location; rejecting an anchor does not move the physical object.

A linear program selects positive interval widths with exact accepted anchors
and adjacent width ratios bounded by `growth`. Incompatible mandatory domain
constraints or grading constraints raise an explanatory error.

Inspect:

```python
print(sim.mesh.report.rejected)           # (rejected request, winner)
print(sim.mesh.report.limited_resolution) # requested spacing below the floor
print(sim.topology.report.staircase_fallbacks)
```

An unresolved port raises an error rather than measuring a different plane.
Geometry rank cannot repair an aperture that has too few retained field cells.

## Conformal topology, sheets, enlargement and stability

Polygon clipping gives fluid areas and boundary lengths. A thin sheet splits
intersected cells into retained regions on both sides. Its terminating tip does
not extend into an artificial opaque wall: a geometrical helper split beyond
the tip is joined by ordinary fluid edges. Splitting adds scalar and vector
degrees of freedom wherever required by the topology.

The signed face-edge incidence matrix `B` is used with its transpose:

\[
 M_q\dot q=-Bp, \qquad M_p\dot p=B^Tq.
\]

`q` is `Hz` for TE or `Ez` for TM; `p` is the tangential vector field oriented
as outward energy flux. Material and geometrical masses are positive. Ordinary
edges have one flux unknown; thin networks can have two. Small connected fluid
faces are merged conservatively using summed area and material mass.
Groups cannot merge the opposing traces of a sheet. Internal edges of a merged
group disappear, while exposed boundary lengths remain in the incidence matrix.

The lossless leapfrog update preserves the modified quadratic energy

\[
 \tfrac12(q^TM_qq+p^TM_pp+\Delta t\,q^TBp).
\]

Enlargement is an area-conservative aggregation approach inspired by enlarged
cell methods, not a reproduction of a particular published ECT stencil. It can
alter local accuracy. Its use alone is not a proof that the original Cartesian
CFL step is valid. The runtime explicitly bounds the spectral radius of
`M_q^-1/2 B M_p^-1 B.T M_q^-1/2` using absolute row sums and applies the selected
Courant factor. Explicit `dt` values above this bound are rejected.

If a fragment cannot enlarge into a connected neighbor, the affected volume
cell falls back to a staircase representation. If a requested `min_dt` still
cannot be met, affected sheets can snap to a staircase path on mesh faces,
while remaining zero thickness. Every fallback records its location or sheet
name and reason. The mesh itself is unchanged. Use `fallback=False` to demand
strict conformal treatment. An impossible budget after fallback raises an error.
`conformal=False` explicitly selects a fully staircase conductor representation.

## Surface models and exact PMC

`SurfaceImpedance` is an opaque, one-sided law. Volumes exclude their interior
fields. A sheet carrying this law has two independent opaque boundary traces.
The scalar passive rational model is

\[
 Z(s)=R+\sum_j r_j\frac{s}{s+p_j},\qquad R,r_j\ge0,\quad p_j>0.
\]

`SurfaceImpedance.good_conductor(conductivity, band, order, tolerance)` fits
`sqrt(s*mu0/conductivity)` with nonnegative Foster terms and rejects fits whose
sampled relative error exceeds the requested tolerance. A constant resistance
is an ideal frequency-independent boundary, not a broadband metal model.

Boundary vector-field and Foster histories are solved together using
trapezoidal integration. TE uses the admittance realization and TM the
impedance realization. PEC and PMC are exact topology limits:
`PEC = SurfaceImpedance(0)` and `PMC = SurfaceImpedance(inf)`. No large material
constants, approximate magnetic cell masks, or `1/inf` arithmetic are used.
PEC sets the electric trace to zero; PMC sets the magnetic trace to zero.

`ThinSheet` is a transmissive symmetric two-port boundary network. In common
and differential current channels its two impedances are `even` and `odd`.
Both sides retain independent field states. Examples:

```python
opaque_wall = SurfaceImpedance.good_conductor(5.8e7, (1e9, 20e9))
resistive_film = ThinSheet.resistive(50)  # ohms per square
metal_film = ThinSheet.conductive(5.8e7, 1e-6, (1e9, 20e9), order=64)
```

The finite-thickness thin-metal model uses the diffusion slab impedances
`Zc*coth(gamma*t/2)` and `Zc*tanh(gamma*t/2)`, represented by passive Foster
expansions. It neglects displacement current inside a good conductor; it is not
a universal optical-film or tensor metasurface model. Its finite-order fit is
checked against the analytic slab over the requested band.

## Ports and S parameters

All ports monitor on every run; any monitored channel may be excited. Multiple
port types and simultaneous coherent source channels, including multiple modes
on one waveguide port, can share a run. A simultaneous drive returns incident
and outgoing waves. A single drive cannot identify a complete multiport S
matrix. `scattering()` independently drives every channel, resets all fields
and memory between runs, and solves `B_out = S A_in` at each frequency.
Insufficient spectral excitation or a singular incoming matrix produces an
invalid bin. Receiver lumped terminals retain their resistance loading.

For TE a lumped terminal is an axis-aligned in-plane voltage line between
`start` and `end`, integrated along resolved electric edges. For TM it is an
out-of-plane terminal at `start`; omit `end`. `invariant_length` is the model's
physical extrusion length. TM terminal voltage is `Ez*invariant_length`.
All ports must share this length for compatible power normalization.
This models an invariant 2D terminal; it does not reproduce a finite 3D coaxial
feed. The source waveform for a lumped port is its Thevenin voltage in volts.

Waveguide ports solve a nonuniform transverse eigenproblem at every requested
DFT frequency. Conductivity and the exact bilinear SIBC response enter the
operator; the leapfrog temporal symbol and longitudinal spatial symbol enter
the propagation constant. Global weighted-overlap assignment and complex phase
transport track modes across frequency. Ambiguous low-overlap matches raise an
error instead of silently switching branches. Below-cutoff channels are marked
invalid for real-power S parameters. Excitation requires a propagating mode
throughout the declared source band. Increase frequency sampling when tracking
or interpolation is insufficient.

The current waveguide implementation requires an axis-aligned interior mesh
plane, one connected aperture with physical side walls, unsplit field cells,
and propagation-invariant transverse materials on the device side. It rejects
enlarged or split cells on the reference plane. Put the port in a straight,
resolved guide continuation. This restriction does not prevent conformal
geometry farther inside the device.

Broadband TF/SF sources synthesize the tracked complex profiles by FFT. The
significant pulse spectrum must lie within the solved band. Modes use one watt
of transported discrete power, including space/time centering factors, and
lumped peak phasors use the matching time-average watt normalization.
TE modes preserve the electric reference orientation when reversing propagation;
their magnetic scalar changes sign. Opposing port normals therefore share a
consistent electric phase convention, which is required for mixed-port reciprocity.
Waveguide source waveform amplitudes are modal amplitudes in `sqrt(W)`.
The Fourier convention is `exp(-i*omega*t)` for accumulation and
`exp(+i*omega*t - i*beta*x)` for forward phasors.

Waveguide ports are matched by default using `VirtualWaveguide()`. A separate
straight field grid clones the device-side dielectric profile and PEC/PMC/SIBC
walls, and ends in longitudinal CPML. At the aperture, one shared flux state
connects the device scalar field to the first auxiliary scalar field through
the same incidence matrix and its transpose. The host's continuation behind
that aperture is disconnected. The auxiliary grid occupies no physical
volume and never appears in geometry plots or NTFF samples. It retains the
whole transverse field grid, allowing outgoing unmonitored and evanescent
content to enter the guide as well as monitored propagating modes.

Excitation uses a broadband modal TF/SF plane inside the auxiliary guide;
passive receiving ports retain their absorber. Waveforms are referenced to
that internal source plane; measured incoming/outgoing amplitudes and S
parameters are referenced to the physical aperture. Independent S runs divide
out the source-to-aperture propagation through their measured incoming matrix.
The source-end guard includes guide transit time. Physical aperture fields
are measured directly, without subtracting a synthetic incident field.

```python
from FDTD_2D import WaveguidePort, VirtualWaveguide
scene.add_port(WaveguidePort("guide", "x", .006, (0, .012), normal=-1,
                virtual_waveguide=VirtualWaveguide(length_cells=40, pml_cells=16)))
```

Finite CPML is an approximation to an infinite guide: near-cutoff modes,
strongly varying dielectric profiles and coarse cells can require more PML
cells and clearance. The regression checks receiver reflection directly,
in addition to checking S parameters, so incoming-matrix calibration cannot
hide a poorly absorbing receiver. `virtual_waveguide=None` explicitly selects
the interior modal source/monitor without an attached matched termination.

## Plane waves and closed NTFF

```python
from FDTD_2D import PlaneWave
scene = Scene()
scene.circle("cylinder", (0, 0), .003, PEC, rank=40)
scene.add_plane_wave(PlaneWave("illumination", axis="x", direction=1))
sim = FDTD2D(scene, mesh_policy=MeshPolicy(.001, 20e9, min_step=.0002, growth=2),
             polarization="TM", frequencies=np.linspace(10e9, 20e9, 21))
run = sim.run(RunControl(1.5e-9),
              {"illumination": GaussianPulse(15e9, 6e-11)})
far = run.far_field(np.linspace(0, 2*np.pi, 181))
```

Plane waves currently support either Cartesian axis and either propagation
direction. The waveform is incident electric-field amplitude in V/m at the
auxiliary 1D launch cell. A mesh-matched 1D background grid supplies incident
fields to a closed TF/SF rectangle. Mask commutators of the same discrete curl
operators inject the total field inside and retain scattered fields outside.
This avoids continuous-plane-wave leakage caused by nonuniform spatial
dispersion. The shell must be homogeneous, Cartesian, and outside CPML.
Oblique incidence is not implemented in this reference.

NTFF is always a closed, simple contour. `ClosedContour.rectangle(bounds)`
generates all four sides; an explicit axis-aligned polygon must repeat its
first vertex at the end. Open contours, omitted sides, unresolved segments,
contours in PML, or contours failing to enclose all geometry and physical
ports are rejected. With plane waves it must also enclose the entire TF/SF
box, placing all samples in the scattered-field region. An automatic domain
adds this monitor by default; `ntff=False` disables measurement. Explicit
domains can pass `ntff=ClosedContour.rectangle((xmin,ymin,xmax,ymax))`.

The transform uses colocated scalar and outward-flux Fourier samples with
the homogeneous outgoing cylindrical Green function. It returns angles in
radians, `scalar_amplitude` with
`q(r,phi) ~ amplitude(phi)*exp(-i*k*r)/sqrt(r)`, and `power_per_radian`.
Raw time DFTs carry a seconds factor, so the latter has units W*s²/radian;
normalizing samples to phasors gives W/radian. TE uses `Hz` and TM uses `Ez`
as the scalar. This continuous-background transform has finite-grid error
and should be checked under mesh refinement. Its analytic regression verifies
the amplitude and sign of an outgoing Hankel wave on two closed contours.

## Outer boundaries and stopping

Domain boundaries default to PEC and can use any supported scalar surface law,
including exact PMC. `PML(width)` adds unsplit CFS-CPML on selected sides;
width is physical. The current CPML requires a homogeneous isotropic uncut
collar. Geometry or ports inside it are rejected. Virtual guides have a
separate uniform-extrusion path with longitudinal CPML and retained side-wall
laws. This does not enable arbitrary curved material-filled PML in the
physical domain.

Every run has a mandatory finite `max_time`; its executed duration never
exceeds that window. Optional `field_tolerance` measures window-maximum field
energy relative to the observed maximum and also checks boundary/PML memory.
`dft_tolerance` compares accumulated complex spectra across complete windows.
DFT convergence requires ports, probe locations or closed NTFF spectra, with windows at least two
cycles of the lowest DFT frequency. Both criteria must pass when both are set.
They must pass `consecutive` times after `min_time` and after every waveform's
declared `end_time`. A zero pre-excitation field never counts as convergence.
These are convergence heuristics rather than error bounds for an arbitrarily
high-Q or weakly excited resonance; select a suitable minimum duration.

`RunResult` includes final scalar/vector fields, time step, executed steps,
stop reason and diagnostic checks. `topology.scalar_grid(result.scalar)` returns
an area-weighted plot array, which intentionally averages distinct split-face
fields in one display cell. Use `topology.faces`, `face_group` and the native
scalar array when inspecting sheet discontinuities.

## Validation and remaining extensions

Focused regression tests cover ranked anchors, real index with permeability,
local fallback, split-sheet topology, exact PEC separation, infinite-SIBC PMC,
lossless modified energy, passive Foster decay, transmissive thin sheets,
CPML absorption, delayed-source guards, field/DFT stopping, complex modal
tracking, matched-receiver reflection, automatic domain generation, closed
NTFF/Hankel equivalence, empty-space TF/SF cancellation, straight-guide phase,
mixed-port reciprocity and analytic sheet
reflection/transmission. The thin-sheet analytic comparison allows finite mesh
and time-discretization error; it is not an arbitrary-shape accuracy guarantee.
The circular PEC cavity fundamental frequency approaches its analytic Bessel
root under refinement: approximately 1.19%, 0.40%, and 0.13% relative error on
12, 24, and 48 cells per domain axis in the focused test.

Bulk anisotropy/ADE dispersion, accelerator kernels, sheet intersections and
general sheet junctions, port planes through split/enlarged cells, and a
conformal material-filled PML and oblique plane waves are explicitly unsupported
in this reference.
The graph/centroid mass approximation still needs systematic mesh-convergence
studies for curved lossy conductors, narrow gaps and oblique films. These are
required before claiming a validated general production solver.

## Research and implementation references

* Reinecke, Thoma and Weiland, *Treatment of Thin, Arbitrary Shaped PEC Sheets
  with FDTD*, 2000. The cell-splitting principle motivates independent field
  states and modified curl topology on either side of a PEC sheet.
  [Proceedings paper, PDF page 40](https://sites.nationalacademies.org/cs/groups/pgasite/documents/webpage/pga_183593.pdf#page=40).
* Bourke et al., *A Conformal Thin Boundary Model for FDTD*, 2018. A useful
  conformal SIBC reference; the implementation here uses graph-based boundary
  traces rather than claiming to reproduce that paper's 3D interpolation stencil.
  [Accepted manuscript](https://eprints.whiterose.ac.uk/id/eprint/131594/1/Bourke_2018_postprint.pdf),
  DOI [10.1109/NEMO.2018.8503410](https://doi.org/10.1109/NEMO.2018.8503410).
* Flintoft et al., *Face-Centered Anisotropic Surface Impedance Boundary
  Conditions in FDTD*, IEEE TMTT 66(2), 643–650, 2018, DOI
  [10.1109/TMTT.2017.2778059](https://doi.org/10.1109/TMTT.2017.2778059).
  The two-sided thin-sheet network must be distinguished from an opaque SIBC.
* [gprMax SIBC theory](https://gprmax.readthedocs.io/en/latest/impedance_surfaces_theory.html)
  and [eigenmode-port theory](https://gprmax.readthedocs.io/en/latest/eigenmode_port_theory.html).
  The local gprMax `impedance_surfaces.py` and `eigenmode_tracking.py` were
  inspected for conventions. `virtual_waveguide.py` and `virtual_waveguide_2d.py`
  were inspected for independent auxiliary-grid coupling and matched receivers.
  This package independently assembles its geometry,
  topology, passive networks and port operators; no gprMax source was copied.
