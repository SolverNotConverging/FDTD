FDTD 2D TEz solver
==================

``FDTD_2D_Hz`` solves the two-dimensional TEz polarization with ``Hz``,
``Ex``, and ``Ey`` on a Yee-staggered grid. Its public workflow mirrors the
TMz solver: named lossy materials, subpixel geometry, native conductors,
CFS-CPML, periodic boundaries, multiple sources, line monitors, power, NF2FF,
animation, persistence, and selectable CPU/GPU backends.

Import and construction
-----------------------

.. code-block:: python

   from FDTD_2D_Hz import FDTD_2D_Hz, Material

   sim = FDTD_2D_Hz(
       x_range=14e-3,
       y_range=14e-3,
       Nx=140,
       Ny=140,
       f_min=50e9,
       f_max=100e9,
       Nt=4000,
       dt=None,
       subpixel=16,
   )
   sim.config("cpu")

Use ``suggest_dx_dt`` to estimate a square-cell resolution and time step from
the maximum material index and frequency.

Yee grid and material mapping
-----------------------------

The field arrays are:

* ``Hz (Nx, Ny)`` at cell centers;
* ``Ex (Nx, Ny + 1)`` on x-directed edges;
* ``Ey (Nx + 1, Ny)`` on y-directed edges.

TEz uses the x/y entries of ``epsilon_r`` and ``sigma_e`` and the z entries of
``mu_r`` and ``sigma_m``. Material starts on cells and is averaged onto the
``Ex`` and ``Ey`` locations. ``Hz`` uses the cell value directly.

Materials and geometry
----------------------

.. code-block:: python

   coating = sim.add_material(
       "coating",
       epsilon_r=(3.0, 4.0, 1.0),
       mu_r=(1.0, 1.0, 1.2),
       sigma_e=(0.02, 0.03, 0.0),
       sigma_m=(0.0, 0.0, 0.001),
   )

   sim.add_rectangle(
       material=coating,
       x_position=(4e-3, 6e-3),
       y_position=(3e-3, 10e-3),
   )
   sim.add_circle(
       material="coating", center=(9e-3, 7e-3), radius=1e-3,
   )
   sim.add_triangle(
       material="PMC",
       vertices=((2e-3, 2e-3), (3e-3, 2e-3), (2.5e-3, 4e-3)),
   )

Floating-point coordinates are interpreted in metres and integers as grid-edge
indices. Curved and partially filled ordinary cells use subpixel area sampling.
PEC and PMC are predefined exact masks. Direct ``ER``, ``MR``, ``sigma_e``,
and ``sigma_m`` shape arguments remain supported for compatibility.

CFS-CPML and periodic boundaries
--------------------------------

.. code-block:: python

   sim.add_PML(
       pml_width=15,
       order=3,
       direction="xy",
       kappa_max=7,
       alpha_max=0.025,
       R0=1e-8,
   )

This is an unsplit complex-frequency-shifted convolutional PML. Recursive
auxiliary fields correct the affected derivatives in both electric and
magnetic updates. Automatic conductivity uses

.. math::

   \sigma_{max}=-\frac{(m+1)\ln(R_0)}{2\eta_0L}.

PML is applied to both ends of every selected axis. ``sim.periodic`` may be
``"x"``, ``"y"``, or ``"xy"``; do not enable PML and periodic wrapping in the
same direction.

Sources
-------

Supported ``add_source`` kinds are:

* ``point``;
* ``line-soft``;
* ``sftf`` for an angled total-field/scattered-field rectangle;
* ``waveguide-x`` and ``waveguide-y`` for modal ports.

.. code-block:: python

   sim.add_source("point", x=7e-3, y=4e-3, amplitude=1.0, is_show=False)
   sim.add_source("line-soft", x=3e-3, y=(4e-3, 10e-3), is_show=False)
   sim.add_source(
       "sftf", x=(25, 115), y=(25, 115),
       angle=0.35, is_show=False,
   )
   sim.add_source(
       "waveguide-y", x=(20, 80), y=20,
       broadband=True,
       frequency_mode_pairs=[
           (60e9, 0),
           (70e9, 0),
           (80e9, 1),
           (90e9, 1),
       ],
       modes_to_show=3,
       is_show=True,
   )

TF/SF requires ``dx == dy``. Modal ports use the local staggered material and
remove conductor-cell degrees of freedom in the FDFD eigenproblem.
For ``broadband=True``, the supplied frequency/index pairs are modal anchors.
The paired index explicitly selects the mode at each anchor. Selected anchor
fields are phase-aligned, then the electric field, magnetic field, and
propagation constant are linearly interpolated onto every real-FFT bin inside
the anchor interval. The dense source spectrum weights those interpolated
fields before inverse-FFT synthesis. The magnetic spectrum includes both the
half-time step and frequency-dependent half-cell propagation phase. At least
two unique positive frequencies are required, and a selected mode with
non-positive ``n_eff`` is rejected as cut off. The mode preview uses frequency
rows and mode-index columns. Mode indices are used exactly as supplied; this
path does not perform automatic cross-frequency mode selection or tracking.

Line monitors and power
-----------------------

.. code-block:: python

   monitor = sim.add_line_monitor(x=110, y=(20, 120), index=10)
   sim.run(record_stride=1, is_include_history=False)

   frequencies = [60e9, 80e9, 100e9]
   power = sim.power_spectrum(monitor, frequencies, source_index=0)
   sim.plot_power_spectrum(power, db=True)

The monitor records the TEz field components needed for signed Poynting flux.
``power_spectrum`` evaluates the requested frequencies directly. FFT-based
alternatives are ``calculate_line_monitor_power_fft`` and
``calculate_source_power_fft``.

NF2FF
-----

.. code-block:: python

   farfield = sim.NF2FF(
       top=10, bottom=20, left=30, right=40,
       freqs=[80e9], nphi=361, src_index=0,
   )
   sim.show_FF(farfield, freq_idx=0, component="Ephi", db=True)

A partial contour is accepted by passing ``None`` for omitted sides. For a
closed scattering calculation, keep the complete contour in homogeneous
background material between the objects and the PML.

Run, animation, and persistence
-------------------------------

``run`` displays tqdm progress and optionally stores full field histories:

.. code-block:: python

   sim.run(record_stride=2, is_include_history=True)
   sim.show_animation(fps=60, dynamic_clim=True)

Save and restore the simulator state with pickle:

.. code-block:: python

   sim.save("tez_run.pkl", include_histories=True)
   restored = FDTD_2D_Hz.load("tez_run.pkl")

States saved before conductivity support are upgraded with zero-loss arrays
when loaded.

Backends
--------

``config("cpu")`` uses the optional Cython curl kernel. ``config("gpu")`` uses
a persistent Numba-CUDA runtime when available, and ``config("python")``
selects the reference loops. The GPU path transfers fields, coefficients,
CFS-CPML arrays, masks, and sparse source descriptions once before the run.
All time-step updates, source injection, monitor sampling, and optional history
recording then remain on-device, followed by one final output synchronization.
No host-device copy occurs inside the time loop.

Use ``is_include_history=False`` with line monitors when GPU memory is limited.
If full histories are requested, they remain on the device until completion;
``record_stride`` controls their size. ``_gpu_transfer_stats`` provides a
post-run diagnostic for source/monitor counts and per-step transfers.

Examples
--------

This directory contains simple-source, TF/SF, single-frequency and broadband
waveguide, far-field, flux, and GPU examples.
