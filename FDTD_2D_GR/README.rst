FDTD_2D_GR: equatorial Schwarzschild light
===========================================

``FDTD_2D_GR`` is a purpose-built polar Yee solver for light near a fixed,
non-rotating Schwarzschild black hole.  It evolves the physical orthonormal TE
components ``Er``, ``Ephi``, and ``Hz`` on the equatorial optical plane.  It is
independent of the repository's Cartesian material solvers.  Schwarzschild
spherical symmetry means this plane represents any orbital plane through the
hole; calling it equatorial simply fixes coordinates.

What is being solved
--------------------

The Schwarzschild line element in isotropic radius ``rho`` is equivalent, for
vacuum Maxwell propagation, to an impedance-matched optical medium with

.. math::

   \epsilon_r(\rho)=\mu_r(\rho)=n(\rho)
   =\frac{\left(1+M/(2\rho)\right)^3}{1-M/(2\rho)}.

This is the exact Schwarzschild equivalent-medium law, not a Newtonian-index
approximation and not ``n**2``.  The local coordinate speed is ``1/n`` in the
solver's geometric units, while the wave impedance remains one.

The polar TE equations are

.. math::

   \partial_t(nH_z)=-\frac{1}{\rho}
   \left[\partial_\rho(\rho E_\phi)-\partial_\phi E_\rho\right],

.. math::

   \partial_t(nE_\rho)=\frac{1}{\rho}\partial_\phi H_z,
   \qquad
   \partial_t(nE_\phi)=-\partial_\rho H_z.

The evolved arrays are complex analytic signals.  Use ``real_hz`` (or
``numpy.real(sim.Hz)`` after ``sim.sync_fields()``) for one instantaneous field
phase, ``abs_hz`` for magnitude, and ``energy`` for the positive wave-energy
view used by the example.

The update is second-order leapfrog on an integral polar Yee mesh.  Azimuth is
exactly periodic.  Smooth equal-rate electric and magnetic loss layers absorb
captured and escaping fields at the radial ends.  They are matched sponges, not
a general curved-space CPML.  ``sponge_reflection`` sizes a flat-coordinate
attenuation profile; optical dwell makes the realized attenuation stronger near
the hole, so it is a tuning target rather than a measured reflection guarantee.

Black-hole landmarks
--------------------

The code uses ``G=c=1`` and takes ``M`` as a coordinate length.  With the
default ``M=1``:

* event horizon: ``rho_h = M/2`` (areal radius ``2M``);
* photon sphere: ``rho_ph = M*(1 + sqrt(3)/2)`` (areal radius ``3M``);
* critical impact parameter: ``b_c = 3*sqrt(3)*M``;
* circular-photon coordinate period: ``T = 6*pi*sqrt(3)*M``.

The isotropic chart covers only ``rho > M/2`` and never crosses the horizon.
The inner sponge represents capture before the coordinate wavelength collapses
near that chart boundary.  Do not use it to infer horizon-scale absorption or
long-time horizon flux.

Quick start
-----------

.. code-block:: python

   from FDTD_2D_GR import FDTD_2D_GR

   sim = FDTD_2D_GR(
       rho_min=0.55,
       rho_max=10.0,
       Nr=320,
       Nphi=640,
       courant=0.60,
   ).config("cpu")
   packet = sim.initialize_orbiting_packet(
       azimuthal_mode=20,
       radial_width=0.35,
       angular_width=0.32,
       direction=+1,
   )
   history = sim.run(
       duration=0.5 * sim.photon_orbit_period,
       record_stride=5,
       store_snapshots=False,
       progress=True,
   )
   sim.plot_snapshot(log_scale=True)
   sim.plot_diagnostics(history)

Run the complete example with

.. code-block:: console

   python FDTD_2D_GR/Example_Photon_Orbit.py

The example is a plain Python-API script like the repository's other 2D
examples.  Edit its constants directly to choose the backend, grid, duration,
and output behavior.  ``SAVE_ANIMATION`` controls whether sampled energy frames
are retained during the run and written to ``photon_packet.mp4`` after time
stepping has finished.  MP4 export requires FFmpeg; set the constant to
``False`` when animation output is not required to avoid the snapshot memory.

Backend selection follows the rest of this repository:

* ``config("python")`` uses the NumPy reference update;
* ``config("cpu")`` uses the optional Cython whole-step kernel;
* ``config("gpu")`` uses persistent Numba-CUDA device fields.

Build the Cython extension from the repository root with
``python setup_cython.py build_ext --inplace``.  If a requested optional
runtime is unavailable, the solver warns and falls back to NumPy.  CUDA fields
stay on the device between steps and are copied back only for recorded
diagnostics, at the end of ``run``, or when ``sync_fields()`` is called.
The current Cython kernel is specialized for ``complex128``; ``complex64`` is
supported by NumPy and CUDA, while ``config("cpu")`` falls back to NumPy for it.

The default run covers half an ideal photon period because that is already
several Lyapunov times.  Set ``NUMBER_OF_ORBITS = 1.0`` for a complete ideal
period, where the capture/escape split is especially obvious.

Interpreting the result
-----------------------

A localized packet must not remain a tidy loop.  In Schwarzschild coordinate
time, the circular photon orbit's linearized Lyapunov exponent equals its
angular frequency, so a sufficiently small radial perturbation grows by
``exp(2*pi)`` (about 535) per ideal revolution while linear theory remains
valid.  In this simulation, the inward/outward split also includes finite-wave
diffraction, numerical dispersion, and launch mismatch because the localized
Gaussian with one carrier frequency is not an exact radial Maxwell eigenmode.
The dashed cyan circle is the exact geometric-optics photon sphere, and the
dashed trajectory in the diagnostics is an ideal reference rather than a claim
that an exact central ray is embedded in the finite packet.

``phi_mean`` is a circular first moment.  Its companion ``phi_coherence`` is
``|<exp(i*phi)>|``: values near one mean a localized angular centroid, while
values near zero mean the packet has spread around the ring and the reported
centroid angle is ill-conditioned.  Keep diagnostic samples less than half an
orbit apart; the solver warns when ``record_stride`` can make ``phi_unwrapped``
alias by more than pi between samples.

This two-dimensional model preserves the correct equatorial null
characteristics in the short-wavelength limit.  It is a cylindrical TE optical
analogue rather than a full 3+1 Maxwell calculation, so it omits out-of-plane
diffraction.  Separately, it uses the test-field approximation: it does not
solve the coupled Einstein-Maxwell system or evolve metric back-reaction.  Kerr
rotation is intentionally absent; adding it would require anisotropic or
bianisotropic constitutive tensors with shift-induced magnetoelectric coupling,
preferably in a horizon-penetrating formulation, rather than a different scalar
``n``.

Resolution guidance
-------------------

Use at least 20--30 azimuthal cells per carrier wavelength, meaning roughly
``Nphi / azimuthal_mode >= 20``.  Resolve the radial Gaussian width with at
least 10--15 cells.  Quantitative studies should repeat the run with ``drho``,
``dphi``, and ``dt`` reduced and should vary both sponge locations.  The
constructor rejects explicit time steps above its local metric-aware CFL
estimate; the default Courant fraction retains a stability margin.

The launch builds ``D`` as a compatible discrete rotated gradient.  Its
electric divergence therefore begins at roundoff.  Spatially varying loss in
the sponges induces an effective charge there, so the reported constraint norm
is measured only in the undamped physical region.

Physical scaling
----------------

Vacuum Schwarzschild propagation is scale-free.  Convert a result for a black
hole of ``mass_solar`` solar masses with

.. code-block:: python

   seconds = sim.to_physical_time(history["time"], mass_solar=4.0e6)
   metres = sim.to_physical_length(sim.photon_sphere_radius, mass_solar=4.0e6)

References
----------

* J. Plebanski, *Electromagnetic Waves in Gravitational Fields*, Physical
  Review 118, 1396 (1960), https://doi.org/10.1103/PhysRev.118.1396
* S. Jia, *Electromagnetic scattering in Schwarzschild space-time: Finite
  difference time domain with Green function method*,
  https://arxiv.org/abs/1804.04298
* K. S. Yee, *Numerical solution of initial boundary value problems involving
  Maxwell's equations in isotropic media*, IEEE TAP 14, 302 (1966),
  https://doi.org/10.1109/TAP.1966.1138693
