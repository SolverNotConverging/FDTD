FDTD 1D solver
==============

``FDTD_1D`` advances the ``Ey`` and ``Hx`` components on a one-dimensional
Yee grid. It supports named anisotropic materials, electric and magnetic loss,
subpixel material assignment, PEC/PMC objects and boundaries, a Gaussian
source, reflection/transmission accumulation, animation, tqdm progress, and an
optional Cython update kernel.

Import
------

.. code-block:: python

   from FDTD_1D import FDTD_1D, Material

Construction
------------

.. code-block:: python

   sim = FDTD_1D(
       z_range=20e-3,
       Nz=500,
       f_max=100e9,
       Nt=5000,
       dt=None,
       subpixel=16,
   )

``z_range`` is in metres. ``Nz`` is the number of material cells, ``Nt`` is
the number of time steps, and ``f_max`` is the highest modeled frequency. If
``dt`` is omitted, the solver applies its CFL and frequency-sampling limits.

Use ``FDTD_1D.suggest_dx_dt(...)`` to obtain an accuracy-oriented cell size and
time step before selecting ``Nz``.

Yee grid
--------

The component arrays are:

* ``Ey.shape == (Nz + 1,)`` at cell faces;
* ``Hx.shape == (Nz,)`` at cell centers.

Material is stored first on ``Nz`` cells, then averaged to the component
locations. For a diagonal material, the 1D polarization uses the y-directed
``epsilon_r`` and ``sigma_e`` entries and the x-directed ``mu_r`` and
``sigma_m`` entries.

Materials and geometry
----------------------

Define a material before adding its geometry:

.. code-block:: python

   sim.add_material(
       "lossy_layer",
       epsilon_r=(2.0, 3.0, 4.0),
       mu_r=(1.0, 1.0, 1.0),
       sigma_e=(0.0, 0.02, 0.0),
       sigma_m=0.0,
   )
   sim.add_object(material="lossy_layer", region=(4e-3, 8e-3))

``region`` can be a physical ``(z_start, z_stop)`` pair or a cell-index slice:

.. code-block:: python

   sim.add_object(material="vacuum", region=slice(0, 50))
   sim.add_object(material="PEC", region=(10e-3, 11e-3))
   sim.add_object(material="PMC", region=slice(350, 360))

Physical spans are clipped to the domain and their boundary cells are mixed
using subpixel midpoint sampling. ``vacuum``, ``PEC``, and ``PMC`` are
predefined. Direct ``ER``, ``MR``, ``sigma_e``, and ``sigma_m`` arguments are
retained for compatibility.

Loss
----

Both electric and magnetic conductivity are included through centered
trapezoidal Yee coefficients. A spatially uniform field with zero curl decays
by the corresponding ``CaEy`` or ``CaHx`` coefficient. Zero conductivity
recovers the lossless update.

Boundaries
----------

Configure both domain ends with ``set_boundary``:

.. code-block:: python

   sim.set_boundary(left="absorbing", right="absorbing")

Accepted values are ``absorbing``, ``PEC``/``electric``, and
``PMC``/``magnetic``. The absorbing option is a first-order outgoing-wave
boundary; this 1D solver does not currently use CPML.

Source and run
--------------

The public source is a soft Gaussian pulse on an interior ``Ey`` location:

.. code-block:: python

   sim.add_source(
       src_position=2e-3,
       amplitude=1.0,
       t0=None,
       tw=None,
       is_show=False,
   )
   sim.run()

``src_position`` may be an integer field index or a floating-point position in
metres. ``run`` displays a tqdm progress bar and fills:

* ``Ey_history`` and ``Hx_history``;
* ``REF_history``, ``TRN_history``, and ``SRC_history``;
* final frequency accumulators ``REF``, ``TRN``, and ``SRC``.

Animation
---------

.. code-block:: python

   sim.show_animation(fps=100)

PEC regions are outlined in dashed yellow and PMC regions in dashed blue.

Cython backend
--------------

The solver imports ``FDTD_1D._cython_kernel_1d`` automatically when it has
been built. Otherwise equivalent Python loops are used. Build from the project
root with:

.. code-block:: console

   python setup_cython.py build_ext --inplace

Complete example
----------------

See ``FDTD_1D_example.py`` in this directory.
