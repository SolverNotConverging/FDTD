# FDTD Studio

Native **C++17 / Qt 6 Widgets / VTK** application for the geometry-first 2D
solver. Python is used only as a separate simulation worker. The modeller,
project management, native menus, VTK viewport and plots are C++.

## Run the built application

From the repository root on Windows:

```powershell
.\gui\build\fdtd-studio.exe
# Or open a saved project:
.\gui\scripts\run.ps1 -Project .\gui\examples\waveguide.fdtd.json
```

The application starts with the PEC-cylinder example. **Examples** also contains
a matched two-port guide, a high-index/SIBC/thin-film network, and a periodic
slotted-guide leaky-wave antenna. Select a
Python interpreter under **Simulation > Python runtime** if the detected one
does not have the solver dependencies. `FDTD_PYTHON` supplies a launch default;
an interpreter explicitly saved in the GUI takes precedence.

```powershell
python -m pip install -r requirements-general-2d.txt
```

Runtime discovery checks the saved setting, `FDTD_PYTHON`, the repository's
`.venv`/`venv`, the available bundled development runtime, then PATH. The worker
adds the repository and an existing `.test-deps` folder to its `PYTHONPATH`.
No PyQt, PySide, Python VTK package, web browser or JavaScript UI is needed.

## Model and run

* Draw rectangles, circles, polygons and zero-thickness sheets using the toolbar,
  or use **Model > Add…** for precise coordinates. Finish polygons with a
  right-click. Esc cancels drawing. The wheel zooms; empty-space dragging pans.
* Select and drag geometry or ports to move them. Double-click their tree entry
  or choose **Edit selected** to change geometry, material or anchor rank.
  Undo/redo covers drawing, edits, moves, deletion, settings and excitations.
* Material choices include PEC, exact infinite-SIBC PMC, isotropic dielectric,
  constant or fitted conducting SIBC, resistive film and finite-thickness thin
  metal. Transmissive films require sheet geometry.
* Add lumped terminals or matched virtual waveguide ports. TM lumped terminals
  are out-of-plane point terminals; TE uses an axis-aligned voltage line.
  Waveguide apertures must lie in a straight resolved cross-section and end on
  physical walls. The port normal points out of the device. All ports use a
  common invariant depth for compatible power normalization.
* **Settings** defines TE/TM polarization, frequency band, pulse, nonuniform
  mesh/CFL limits and stopping criteria. Coordinates are **mm**, frequency
  **GHz**, pulse width **ps**, and run duration **ns**. Physics is converted to
  SI by the worker. The simulation domain, PML and closed measurement boxes
  are generated after object placement.
* **Generate mesh / F6** compiles without running. **Show mesh** displays the
  actual nonuniform X/Y lines over the geometry and can hide them while editing.
  These are simulation lines, separate from the adaptive drawing grid.
  The model view shows domain,
  closed NTFF, optional TF/SF and physical bounds. The log reports dt, DOFs,
  rejected anchors, enlargement and staircase fallbacks. VTK shows retained
  conformal polygons; enable **Cell edges** to inspect them.
* **Run / F5** starts a worker process. A full S study independently drives every
  monitored port mode. A coherent run uses checked transmitting channels and
  their amplitudes; all ports receive. The UI stays responsive and shows live
  field snapshots, progress and errors. Geometry and excitation edits are
  disabled during a run. **Stop** cancels cooperatively; a worker that remains
  busy compiling is terminated after three seconds. Cancelled runs do not
  publish a completed result.

The full guide pulse must fit the solved frequency band and remain propagating;
otherwise the solver reports an error. **Simulation log** preserves that message
and the Python traceback for diagnosis.

**Port modes** is available immediately after mesh generation. Select a guide,
tracked mode and anchor frequency to inspect its complex scalar field (`Ez`/`Hz`)
or power-conjugate tangential field (`Ht`/`Et`). Magnitude, real, imaginary and
phase views use the solver's actual tracked bank, preserving its phase across
frequency. Dispersion plots show every tracked mode's phase constant, attenuation,
effective index or tracking overlap. Frequency anchors are the solved frequency
samples from Settings. Propagating modes use 1 W normalization at the configured
invariant depth; evanescent modes retain eigenvector scaling. The displayed
dispersion belongs to the straight port cross-section, including spatial/time
discretization; it is not a Bloch dispersion solve of the complete antenna.

Add **Model > Add field monitor** or draw a rectangle with the **Monitor** tool.
Set a comma-separated, increasing frequency list in GHz. Monitors are measurement
regions: they do not change material, add mesh anchors, or expand the automatic
domain. A region must fit the generated domain and contain retained field cells.
Its frequencies can differ from the port/S-parameter frequency anchors, provided
they remain below the time-step Nyquist limit. Monitor drawing, movement, editing,
save/load and deletion support undo/redo.

## Inspect results

**Mesh & fields** provides permittivity, a peak-energy scalar snapshot and final
scalar fields for every drive. VTK retains separate sheet-side field values;
it does not average distinct conformal states into a Cartesian display cell.
TE displays `Hz` (A/m) and TM `Ez` (V/m). Field snapshots exclude virtual guides.
VTK interaction supports zoom/pan and the toolbar's **Fit** resets the view.

**Frequency fields** selects a run, regional monitor and frequency. Displays
the complex scalar DFT as magnitude, real, imaginary or phase, on polygons clipped
to that monitor's bounds. Separate thin-sheet sides remain separate. The solver
accumulates every time step with `exp(-i omega t) * dt`, including monitor spectra
in enabled DFT convergence checks. These are raw Fourier integrals, with units
V*s/m for `Ez` and A*s/m for `Hz`, rather than steady-state amplitudes normalized
to source power. The stored field is total inside TF/SF and scattered outside.

Choose **Harmonic animation** and **Play** to cycle through
`Re(DFT * exp(+i phase))`; **Pause** and the phase slider inspect individual phases.
The color range stays fixed through the cycle. Playback is slowed to about 0.37
cycles/s; the selected physical period is shown in ps. Animation reconstructs
that frequency's harmonic spatial pattern from the complex DFT and does not replay
the broadband transient. It also works after reopening saved results. **PNG**
exports the currently displayed phase.

**S parameters** selects an incident channel and plots every outgoing channel.
Magnitude in dB, wrapped phase, real and imaginary representations are available.
Invalid frequency bins remain gaps. Hover for values, wheel to zoom the frequency
axis, and double-click to reset. CSV exports retain invalid gaps.

**Far field** selects a run and frequency. Available plots are relative power
in dB, power per incident watt for a single excited port, raw Fourier power,
and scalar phase. Angles are degrees measured counterclockwise from +X.
Raw Fourier power carries units W*s²/radian. Port normalization divides by the
measured incoming wave and yields W/radian per incident W. Coherent multi-source
and plane-wave runs provide relative patterns/raw spectra; a single incident
port power normalization is unavailable. An exactly nonradiating guide can have
no finite relative pattern. The underlying continuous-background 2D NTFF has
finite-grid error and retains the solver's validation limits.

CSV and PNG buttons export the displayed plots. Field PNG uses VTK's framebuffer.
**File > Open simulation results** reopens a saved run without recomputation,
including from a moved run directory. **File > Open run folder** opens all exports.

Each job is saved under `gui/runs/<timestamp>-<id>/`:

* `project.fdtd.json`: exact input snapshot, version 1, millimetre units.
* `mesh.json`, `mesh.vtu`: generated grid, reports and native conformal field mesh.
* `results.json`: curves, complex waves, far fields and convergence diagnostics.
* `sparameters.npz` and `sparameters.csv`: full matrix studies.
* `fields_and_far_field.npz`, `far_field.csv`: native fields and angular spectra.
* `field_peak_*.vtu`, `field_final_*.vtu`: persistent inspection snapshots.
* `monitor_<run>_<region>_<frequency>.vtu`: clipped real, imaginary, magnitude and
  phase cell arrays. `monitor_<run>_<region>.npz` retains complex DFTs, frequencies,
  native group indices/centers and SI region bounds.

## Leaky-wave example

**Examples > Periodic leaky-wave antenna** opens
[`leaky-wave.fdtd.json`](examples/leaky-wave.fdtd.json). It models a 100 mm PEC
parallel-plate guide with a 7 mm aperture, seven 2 mm roof slots at 10 mm pitch,
a 0.5 mm roof, and matched feed/termination ports. The 24–40 GHz band stays below
the second straight-guide mode cutoff; the default study computes the full
two-port S matrix. The regional monitor includes the guide and the radiating
air above the slots at 28, 32 and 36 GHz.

Run the study, inspect feed-mode dispersion, then compare the far field and
animated monitor fields across frequency. Periodic slots couple guided waves to
radiating spatial harmonics. This is an educational 2D model with finite-grid
error, not a reproduced 3D hardware design or an optimized antenna.
The periodic-slot mechanism is described in the primary paper
[Leaky Mode Generation by means of All-Metal Corrugated Slotted Waveguides](https://arxiv.org/abs/2504.04229).
That paper also adds corrugations; this example uses a simpler slotted guide.

The supplied example was checked on its default 492 × 117 generated mesh
(0.530697 ps time step). Both independent port drives converged at 2.76706 ns.
For the feed drive, the sampled upper-half-space main beams were 114°, 101°
and 91° at 28, 32 and 36 GHz respectively, with radiated fractions about
1.69%, 1.88% and 2.99%. These are finite-grid results for this configuration.
The integrated far-field power agrees with the missing two-port power to
within 0.1 percentage point at those three frequencies.

The completed local run can be inspected without rerunning:

```powershell
.\gui\build\fdtd-studio.exe --results gui/runs/leaky-wave-final/results.json
```

The run directory is generated/ignored data; other checkouts need to run the
example first.

Project files are data only. The bridge does not evaluate code from projects.

## Build on Windows

Requires an x64 MSVC C++ compiler/Windows SDK (Visual Studio 2022 or later),
Python with pip, and network access for the initial downloads. Dependencies
are installed inside the repository; no global Qt/VTK installation is changed.

```powershell
.\gui\scripts\bootstrap.ps1 -Python C:\path\to\python.exe
```

This installs Qt 6.8.3 qtbase, retains its matching source/license archive,
builds VTK 9.5.2 with Qt integration, compiles the GUI and tests, and deploys
dynamic libraries and vendor notices beside the executable.
Subsequent rebuilds only need:

```powershell
cmd /c gui\scripts\build.cmd
```

With separately installed Qt/VTK on Windows, Linux or macOS, configure directly:

```sh
cmake -S gui -B gui/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/path/to/Qt;/path/to/VTK"
cmake --build gui/build --parallel
```

The native app requires a compatible OpenGL driver for VTK. The local developer
build retains a source-tree path for the worker and examples; standalone
relocatable packaging is not implemented. Qt and VTK are dynamically linked;
see [third-party notices](THIRD_PARTY.md).

## Verify

```powershell
.\.gui-tools\cmake\data\bin\ctest.exe --test-dir gui/build --output-on-failure
python -m pytest tests/test_gui_backend.py tests/test_general_2d.py -q
```

The native tests cover persistence, undo, physical coordinate transforms,
excitation references, nonuniform mesh visibility, monitor editing, modal anchor
selection, invalid project handling and plot exports. The Python
bridge tests cover units, generated closed boxes, native conformal field files,
read-only progress callbacks, every-step regional DFTs, clipped complex field
exports, tracked modal banks and JSON/CSV/NPZ output.

The executable also provides process-level test options:

```powershell
.\gui\build\fdtd-studio.exe --project gui/examples/waveguide.fdtd.json --run --exit-after-run --smoke-test gui/build/artifacts/waveguide
.\gui\build\fdtd-studio.exe --results path/to/results.json --smoke-test gui/build/artifacts/reopen
```

`--smoke-test` captures native model, field, S-parameter, far-field, port-mode and
frequency-field tabs, plus 0/90/180-degree monitor phases when available, and
exits. `--cancel-after-ms 500` tests the stop path with `--run --exit-after-run`.
Exit codes are 0 for success, 2/3 for failed input/result loading, 4 for a failed
or cancelled job, and 5 for a failed standalone rendering capture.
