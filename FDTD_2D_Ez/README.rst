FDTD 2D TMz solver
==================

``FDTD_2D_Ez`` solves the two-dimensional TMz polarization with ``Ez``,
``Hx``, and ``Hy`` on their exact Yee locations. It supports named lossy
materials, subpixel geometry, PEC/PMC regions, CFS-CPML, periodic boundaries,
several source types, line monitors, power analysis, 2D NF2FF, animation,
pickle state persistence, and Python/Cython/Numba-CUDA backends.

Import and construction
-----------------------

.. code-block:: python

   from FDTD_2D_Ez import FDTD_2D_Ez, Material

   sim = FDTD_2D_Ez(
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

``suggest_dx_dt`` provides a square-cell recommendation and stable time step.

Yee grid and material mapping
-----------------------------

The field arrays are:

* ``Ez (Nx + 1, Ny + 1)`` on nodes;
* ``Hx (Nx + 1, Ny)`` on x-directed edges;
* ``Hy (Nx, Ny + 1)`` on y-directed edges.

TMz uses the z entries of ``epsilon_r`` and ``sigma_e`` and the x/y entries of
``mu_r`` and ``sigma_m``. Cell material is averaged onto the corresponding
Yee locations before coefficients are initialized.

Materials and geometry
----------------------

.. code-block:: python

   slab = sim.add_material(
       "slab",
       epsilon_r=(3.0, 3.0, 4.0),
       mu_r=(1.1, 1.2, 1.0),
       sigma_e=(0.0, 0.0, 0.02),
       sigma_m=(0.001, 0.002, 0.0),
   )

   sim.add_rectangle(
       material=slab,
       x_position=(4e-3, 6e-3),
       y_position=(3e-3, 10e-3),
   )
   sim.add_circle(
       material="slab", center=(9e-3, 7e-3), radius=1e-3,
   )
   sim.add_triangle(
       material="PEC",
       vertices=((2e-3, 2e-3), (3e-3, 2e-3), (2.5e-3, 4e-3)),
   )

Positions may be floating-point metres or integer grid-edge indices. Ordinary
shapes use subpixel area sampling. PEC and PMC use exact conductor masks and do
not rely on extreme material values. Direct ``ER``, ``MR``, ``sigma_e``, and
``sigma_m`` shape arguments remain available for older scripts.

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

The solver uses unsplit complex-frequency-shifted convolutional PML with
recursive auxiliary fields. The automatically selected peak conductivity uses
the standard natural-log expression

.. math::

   \sigma_{max}=-\frac{(m+1)\ln(R_0)}{2\eta_0L}.

Selected directions receive matched PMLs on both lower and upper boundaries.
Set ``sim.periodic`` to ``"x"``, ``"y"``, or ``"xy"`` when periodic curl
boundaries are required. Do not combine a periodic direction with a PML in the
same direction.

Sources
-------

``add_source`` accepts:

* ``point`` -- soft point excitation;
* ``line-soft`` -- soft horizontal or vertical line;
* ``sftf`` -- angled total-field/scattered-field rectangle;
* ``waveguide-x`` -- modal port propagating toward +x;
* ``waveguide-y`` -- modal port propagating toward +y.

Examples:

.. code-block:: python

   sim.add_source("point", x=7e-3, y=4e-3, amplitude=1.0, is_show=False)
   sim.add_source("line-soft", x=3e-3, y=(4e-3, 10e-3), is_show=False)
   sim.add_source(
       "sftf", x=(25, 115), y=(25, 115),
       angle=0.35, is_show=False,
   )

The TF/SF source requires square cells. Waveguide sources solve a reduced
staggered FDFD eigenproblem using the local material and conductor masks.

Line monitors and power
-----------------------

.. code-block:: python

   monitor = sim.add_line_monitor(x=110, y=(20, 120), index=10)
   sim.run(record_stride=1, is_include_history=False)

   frequencies = [60e9, 80e9, 100e9]
   power = sim.power_spectrum(monitor, frequencies, source_index=0)
   sim.plot_power_spectrum(power, db=True)

Monitor IDs are stable non-negative integers. ``power_spectrum`` evaluates a
direct DFT at the requested frequencies and integrates signed Poynting flux
along the complete line. FFT convenience methods are also available:
``calculate_line_monitor_power_fft``, ``calculate_source_power_fft``, and
``plot_fft_results``.

NF2FF
-----

Create line monitors on the desired equivalence contour and pass their IDs:

.. code-block:: python

   farfield = sim.NF2FF(
       top=10, bottom=20, left=30, right=40,
       freqs=[80e9], nphi=361, src_index=0,
   )
   sim.show_FF(farfield, freq_idx=0, component="Etheta", db=True)

Any side may be ``None`` if at least one monitor is supplied. For a physically
closed radiation transform, place the contour in homogeneous material,
outside all scatterers and inside the PML.

Run, animation, and persistence
-------------------------------

``run`` displays tqdm progress. Set ``is_include_history=False`` when full
field animation history is unnecessary. Otherwise:

.. code-block:: python

   sim.run(record_stride=2)
   sim.show_animation(fps=60, dynamic_clim=True)

The full simulator state can be saved and restored with pickle:

.. code-block:: python

   sim.save("tmz_run.pkl", include_histories=True)
   restored = FDTD_2D_Ez.load("tmz_run.pkl")

Backends
--------

``config("cpu")`` uses Cython curl kernels when built and otherwise falls back
to Python. ``config("gpu")`` uses a persistent Numba-CUDA runtime when
available. It uploads simulation state once, advances curl, CFS-CPML, lossy
material updates, conductor masks, sparse sources, monitor gathering, and
optional history recording on the device, and copies final/output arrays back
after the run. There are no host-device transfers inside the GPU time loop.
``config("python")`` forces the reference loops.

For memory-efficient GPU work, use line monitors and
``is_include_history=False``. Full-field histories remain device-resident until
the run completes and therefore must fit in GPU memory; increase
``record_stride`` to reduce their size. After a GPU run,
``_gpu_transfer_stats`` reports the number of compiled source events and
monitor points and confirms that per-step transfer counts are zero.

Examples
--------

This directory contains simple-source, TF/SF, waveguide, far-field, flux, and
GPU examples.
