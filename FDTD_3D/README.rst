FDTD 3D solver
==============

``FDTD_3D`` is a full-vector Cartesian Yee solver with named anisotropic lossy
materials, block/cylinder/sphere geometry, exact PEC/PMC masks, 3D CFS-CPML,
soft point/line/plane sources, plane monitors, HDF5 persistence, power spectra,
surface-equivalence NF2FF, 3D dB far-field plotting, tqdm progress, a compiled
whole-run Cython backend, and a device-resident Numba-CUDA backend.

Import and construction
-----------------------

.. code-block:: python

   from FDTD_3D import FDTD_3D, Material

   sim = FDTD_3D(
       x_range=100e-3,
       y_range=100e-3,
       z_range=100e-3,
       Nx=100,
       Ny=100,
       Nz=100,
       f_min=6e9,
       f_max=10e9,
       Nt=2000,
       dt=None,
       subpixel=4,
   )
   sim.config("cpu")

``suggest_dx_dt`` returns a cubic cell size and a stable time step based on the
highest frequency and maximum refractive index. Three-dimensional histories can
be large; estimate monitor and field memory before starting a production run.

Yee grid
--------

The component arrays are:

* ``Ex (Nx, Ny + 1, Nz + 1)``;
* ``Ey (Nx + 1, Ny, Nz + 1)``;
* ``Ez (Nx + 1, Ny + 1, Nz)``;
* ``Hx (Nx + 1, Ny, Nz)``;
* ``Hy (Nx, Ny + 1, Nz)``;
* ``Hz (Nx, Ny, Nz + 1)``.

Diagonal constitutive and conductivity tensors are volume-rasterized on voxels
and averaged to the matching edges or faces. All six lossy updates use centered
trapezoidal coefficients.

Materials and geometry
----------------------

.. code-block:: python

   dielectric = sim.add_material(
       "dielectric",
       epsilon_r=(2.2, 2.3, 2.4),
       mu_r=1.0,
       sigma_e=2e-4,
       sigma_m=0.0,
   )

   sim.add_block(
       dielectric,
       x=(13e-3, 17e-3),
       y=(13e-3, 17e-3),
       z=(9e-3, 12e-3),
   )
   sim.add_cylinder(
       "dielectric", center=(30e-3, 30e-3, 30e-3),
       radius=3e-3, height=12e-3, axis="z",
   )
   sim.add_sphere("PEC", center=(50e-3, 50e-3, 50e-3), radius=4e-3)

Spans and centers accept integer indices or floating-point metres. Blocks use
half-open spans. Curved ordinary geometry uses subpixel volume sampling;
``subpixel=4`` means 4 samples along each voxel axis. PEC and PMC are predefined
and applied as exact Yee-component constraints.

CFS-CPML
--------

.. code-block:: python

   sim.add_PML(
       pml_width=15,
       order=3,
       direction="xyz",
       kappa_max=7.0,
       alpha_max=0.05,
       R0=1e-8,
   )

The full 3D boundary is an unsplit complex-frequency-shifted convolutional PML.
It maintains twelve recursive auxiliary arrays: two corrected derivatives for
each electric and magnetic component. Profiles are sampled separately at node
and cell coordinates so they follow the Yee staggering.

Automatic peak conductivity is calculated independently for each selected
axis using

.. math::

   \sigma_{max}=-\frac{(m+1)\ln(R_0)}{2\eta_0L}.

``pml_width`` may be a cell count, a physical thickness, or a length-three
sequence. The width on an enabled axis must be less than half its grid count.

Sources
-------

Point, line, and plane sources are additive soft electric sources. Their
polarization can be ``x``, ``y``, or ``z``. A tuple denotes a half-open span;
the requested source kind must have zero, one, or two spans respectively.

.. code-block:: python

   sim.add_source("point", x=50, y=50, z=25, polarization="z")
   sim.add_source("line", x=(30, 70), y=50, z=25, polarization="x")
   sim.add_source(
       "plane", x=(30, 70), y=(30, 70), z=25,
       polarization="x", amplitude=1.0,
       f_min=6e9, f_max=10e9,
   )

With ``f_min=None`` the waveform is Gaussian. Equal nonzero ``f_min`` and
``f_max`` select a smoothly ramped continuous tone; a frequency interval uses
a sinusoid under a Gaussian envelope.

Plane monitors
--------------

Monitors sample all six staggered components at voxel centers. Define a plane
with a stable ID and optional automatic HDF5 output:

.. code-block:: python

   monitor = sim.add_plane_monitor(
       axis="z",
       position=30,
       first=(20, 80),
       second=(20, 80),
       index=5,
       normal="+",
       save_path="output/z30.h5",
   )

``first`` and ``second`` describe the two transverse spans. The monitor normal
controls the sign of Poynting power and equivalent surface currents.

Run and progress
----------------

.. code-block:: python

   results = sim.run(
       steps=None,
       record_stride=2,
       reset=False,
       progress=True,
       progress_desc="3D scattering",
   )

The default tqdm progress display works with every backend. The Cython loop is
advanced in coarse chunks while preserving source timing and monitor-stride
alignment. ``record_stride`` reduces monitor storage, not the FDTD time step.

``reset_fields`` clears fields and time state. ``step`` advances one step for
interactive use.

HDF5 monitor persistence
------------------------

Monitor files use gzip-compressed HDF5 and contain all six fields, sample
times, physical coordinates, surface normal, plane shape, and geometry metadata.
Files can be analyzed without rerunning the simulation:

.. code-block:: python

   sim.save_plane_monitor(monitor, "output/z30_copy.h5")
   saved = sim.load_plane_monitor("output/z30.h5", register=False)

   sim.plot_plane_monitor(saved, component="Ex", time_index=-1)
   sim.plot_plane_monitor(
       saved, component="Ex", frequency=8e9,
       representation="magnitude", window="hann",
   )

   figure, animation = sim.animate_plane_monitor(
       "output/z30.h5", component="Ex", frame_stride=2,
   )

``plot_plane_monitor`` and ``animate_plane_monitor`` accept a registered ID, a
loaded monitor dictionary, or an HDF5 path. Available quantities are
``Ex/Ey/Ez``, ``Hx/Hy/Hz``, vector magnitudes ``E`` and ``H``, and normal
Poynting flux ``Snormal``. Animation output can be GIF through Pillow or another
format supported by an installed Matplotlib writer.

Power spectrum
--------------

.. code-block:: python

   frequencies = [6e9, 8e9, 10e9]
   power = sim.power_spectrum(
       monitor, frequencies, source_index=0, window="hann",
   )
   sim.plot_power_spectrum(power, db=True)

The calculation applies the Yee half-time-step phase correction to magnetic
fields, forms the complex time-average Poynting vector, and integrates signed
power over the plane. Passing ``source_index`` normalizes by the selected source
spectrum.

NF2FF and 3D far-field plot
---------------------------

Use six outward-facing planes to form a closed equivalence box:

.. code-block:: python

   box = sim.add_nf2ff_box(
       x=(20, 80), y=(20, 80), z=(20, 80), start_index=10,
   )
   sim.run(record_stride=1)

   farfield = sim.NF2FF(
       box,
       freqs=[8e9],
       source_index=0,
       window="hann",
   )
   sim.plot_nf2ff(farfield, db=True, db_floor=-40.0)

``plot_nf2ff`` renders a three-dimensional radiation surface whose radius and
color represent normalized radiation intensity in dB. ``plot_nf2ff_cut``
produces an optional polar cut. Keep the equivalence box closed, in homogeneous
background material, outside every scatterer, and inside the PML.

Backends
--------

``config("cpu")`` uses ``_cython_kernel_3d.run_fdtd`` when the extension is
available. This compiled function advances the main update/source/monitor loop.
``config("gpu")`` selects the Numba-CUDA runtime when a CUDA device is
available. It keeps all six fields, all twelve CPML auxiliaries, update
coefficients, precomputed soft-source samples, and packed plane-monitor output
on the device throughout the time loop. There are no field or monitor-array
transfers per time step; final mutable state and recorded monitor data are
copied back once after synchronization. Increase ``record_stride`` when a
large monitor history would consume too much device memory.
``config("python")`` selects the equivalent NumPy reference implementation.

.. code-block:: python

   sim.config("gpu")
   sim.run(record_stride=4, progress=True)

If Numba-CUDA or a CUDA device is unavailable, ``config("gpu")`` emits a
warning and selects the NumPy fallback.

Build the extension from the project root:

.. code-block:: console

   python setup_cython.py build_ext --inplace

Complete example
----------------

``Example_3D.py`` is the CPU/Cython case and ``Example_3D_GPU.py`` selects the
device-resident CUDA backend. Both default to a ``100 x 100 x 100`` grid and
use the same lossy-dielectric/PEC scatterer, external plane excitation, closed
NF2FF box, HDF5 plane monitor, power calculation, and 3D dB far-field plot.
The shared setup makes their saved output directly comparable.

.. code-block:: console

   python FDTD_3D/Example_3D.py --no-show
   python FDTD_3D/Example_3D_GPU.py --no-show

The examples reject grids smaller than 100 cells per axis. Use ``--steps`` and
``--record-stride`` to balance spectral duration against monitor memory,
``--output-dir`` to select a destination, and ``--animate`` to additionally
write the recorded field as a GIF. The default stride is eight, reducing the
million-cell examples' plane-monitor footprint while retaining sufficient
sampling bandwidth for their 6--10 GHz excitation.
