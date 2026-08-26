"""Equatorial TE Maxwell FDTD on a fixed Schwarzschild background.

The solver uses isotropic Schwarzschild coordinates and the equivalent-medium
form of vacuum Maxwell theory.  In geometric units (``G = c = 1``), the
effective relative permittivity and permeability are both

    n(rho) = (1 + M / (2 rho))**3 / (1 - M / (2 rho)).

The numerical grid is polar and Yee staggered.  The arrays store physical
orthonormal polar components: ``Hz`` lives at cell centres, ``Er`` at
azimuthal faces, and ``Ephi`` at radial faces.  Azimuth is exactly periodic.
Smooth matched electric/magnetic sponges and radial characteristic conditions
truncate the exterior domain.

This is a fixed-background wave solver: the electromagnetic field does not
back-react on the black hole.  Its two-dimensional equatorial reduction has
the correct Schwarzschild optical characteristics, including the photon orbit,
but it does not model diffraction out of the equatorial plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings

import numpy as np

try:
    from . import _cython_kernel_gr
except ImportError:
    _cython_kernel_gr = None


@dataclass(frozen=True)
class SchwarzschildGeometry:
    """Schwarzschild geometry expressed in isotropic radial coordinates.

    Parameters
    ----------
    mass:
        Geometric mass ``M = G M_physical / c**2`` in the solver's coordinate
        length units.  The default ``M=1`` is the usual scale-free convention.
    """

    mass: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.mass) or self.mass <= 0.0:
            raise ValueError("mass must be a finite positive geometric length.")

    @property
    def horizon_isotropic_radius(self) -> float:
        return 0.5 * self.mass

    @property
    def horizon_areal_radius(self) -> float:
        return 2.0 * self.mass

    @property
    def photon_sphere_isotropic_radius(self) -> float:
        return self.mass * (1.0 + 0.5 * np.sqrt(3.0))

    @property
    def photon_sphere_areal_radius(self) -> float:
        return 3.0 * self.mass

    @property
    def critical_impact_parameter(self) -> float:
        return 3.0 * np.sqrt(3.0) * self.mass

    @property
    def photon_orbit_angular_velocity(self) -> float:
        return 1.0 / self.critical_impact_parameter

    @property
    def photon_orbit_period(self) -> float:
        return 2.0 * np.pi * self.critical_impact_parameter

    def lapse(self, rho: np.ndarray | float) -> np.ndarray:
        rho_array = np.asarray(rho, dtype=float)
        self._validate_exterior_radius(rho_array)
        compactness = self.mass / (2.0 * rho_array)
        return (1.0 - compactness) / (1.0 + compactness)

    def refractive_index(self, rho: np.ndarray | float) -> np.ndarray:
        """Return the exact isotropic-coordinate optical index ``eps=mu=n``."""

        rho_array = np.asarray(rho, dtype=float)
        self._validate_exterior_radius(rho_array)
        compactness = self.mass / (2.0 * rho_array)
        return (1.0 + compactness) ** 3 / (1.0 - compactness)

    def areal_radius(self, rho: np.ndarray | float) -> np.ndarray:
        rho_array = np.asarray(rho, dtype=float)
        self._validate_exterior_radius(rho_array)
        compactness = self.mass / (2.0 * rho_array)
        return rho_array * (1.0 + compactness) ** 2

    def optical_circumference_radius(self, rho: np.ndarray | float) -> np.ndarray:
        """Return ``rho*n(rho)``, whose minimum is the photon sphere."""

        rho_array = np.asarray(rho, dtype=float)
        return rho_array * self.refractive_index(rho_array)

    def _validate_exterior_radius(self, rho: np.ndarray) -> None:
        if np.any(~np.isfinite(rho)):
            raise ValueError("rho contains non-finite values.")
        if np.any(rho <= self.horizon_isotropic_radius):
            raise ValueError(
                "isotropic Schwarzschild coordinates require rho > M/2 "
                "outside the event horizon."
            )


class FDTD_2D_GR:
    """Polar TEz FDTD dedicated to equatorial Schwarzschild light propagation.

    The solver uses geometric units with ``G=c=1``.  Electric fields are
    normalized conventionally and ``Hz`` denotes ``eta0 * H_z``, so vacuum
    impedance is one and all three evolved fields have the same numerical unit.

    Parameters
    ----------
    rho_min, rho_max:
        Inner and outer isotropic radii.  ``rho_min`` must lie outside ``M/2``.
    Nr, Nphi:
        Radial and azimuthal cell counts.
    mass:
        Schwarzschild geometric mass ``M`` in coordinate length units.
    courant:
        Fraction of the spatially varying polar-grid CFL estimate.
    dt:
        Optional explicit time step.  Values above the CFL limit are rejected.
    inner_sponge_width, outer_sponge_width:
        Radial matched-loss layer widths.  Pass zero to disable a layer.
    sponge_reflection:
        Nominal one-pass amplitude attenuation used to size each sponge.
    radial_boundary:
        ``"characteristic"`` for one-way impedance conditions or ``"pec"``
        for closed radial walls.  The default characteristic boundary should
        normally be combined with both sponges.
    """

    G_SI = 6.67430e-11
    C_SI = 299_792_458.0
    SOLAR_MASS_KG = 1.98847e30

    def __init__(
        self,
        rho_min: Optional[float] = None,
        rho_max: float = 10.0,
        Nr: int = 320,
        Nphi: int = 512,
        mass: float = 1.0,
        courant: float = 0.65,
        dt: Optional[float] = None,
        inner_sponge_width: Optional[float] = None,
        outer_sponge_width: Optional[float] = None,
        sponge_reflection: float = 1.0e-8,
        sponge_order: int = 4,
        radial_boundary: str = "characteristic",
        dtype=np.complex128,
    ) -> None:
        self.geometry = SchwarzschildGeometry(float(mass))
        self.mass = self.geometry.mass

        if rho_min is None:
            rho_min = 0.55 * self.mass
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        self.Nr = int(Nr)
        self.Nphi = int(Nphi)

        if self.Nr < 8:
            raise ValueError("Nr must be at least 8.")
        if self.Nphi < 16:
            raise ValueError("Nphi must be at least 16.")
        if self.rho_min <= self.geometry.horizon_isotropic_radius:
            raise ValueError("rho_min must be strictly outside the horizon M/2.")
        if not np.isfinite(self.rho_max) or self.rho_max <= self.rho_min:
            raise ValueError("rho_max must be finite and greater than rho_min.")
        if not np.isfinite(courant) or not 0.0 < courant <= 1.0:
            raise ValueError("courant must satisfy 0 < courant <= 1.")
        if not isinstance(sponge_order, (int, np.integer)) or sponge_order < 1:
            raise ValueError("sponge_order must be a positive integer.")
        if not np.isfinite(sponge_reflection) or not 0.0 < sponge_reflection < 1.0:
            raise ValueError("sponge_reflection must lie strictly between zero and one.")
        if radial_boundary not in {"characteristic", "pec"}:
            raise ValueError("radial_boundary must be 'characteristic' or 'pec'.")

        resolved_dtype = np.dtype(dtype)
        if resolved_dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
            raise ValueError(
                "The solver uses complex analytic fields; dtype must be complex64 "
                "or complex128."
            )
        self.dtype = resolved_dtype
        self.radial_boundary = radial_boundary
        self.sponge_reflection = float(sponge_reflection)
        self.sponge_order = int(sponge_order)

        self.drho = (self.rho_max - self.rho_min) / self.Nr
        self.dphi = 2.0 * np.pi / self.Nphi
        self.rho_faces = np.linspace(self.rho_min, self.rho_max, self.Nr + 1)
        self.rho_centers = 0.5 * (self.rho_faces[:-1] + self.rho_faces[1:])
        self.phi_faces = np.arange(self.Nphi, dtype=float) * self.dphi
        self.phi_centers = (np.arange(self.Nphi, dtype=float) + 0.5) * self.dphi

        self.n_hz = self.geometry.refractive_index(self.rho_centers)
        self.n_er = self.n_hz.copy()
        self.n_ephi = self.geometry.refractive_index(self.rho_faces)

        local_rate = np.sqrt(
            (1.0 / self.drho) ** 2
            + (1.0 / (self.rho_centers * self.dphi)) ** 2
        ) / self.n_hz
        self.dt_cfl = 1.0 / float(np.max(local_rate))
        if dt is None:
            self.dt = float(courant) * self.dt_cfl
        else:
            self.dt = float(dt)
            if not np.isfinite(self.dt) or self.dt <= 0.0:
                raise ValueError("dt must be finite and positive.")
            if self.dt > self.dt_cfl * (1.0 + 1.0e-12):
                raise ValueError(
                    f"dt={self.dt:g} exceeds the estimated GR polar-grid CFL "
                    f"limit {self.dt_cfl:g}."
                )
        self.courant = self.dt / self.dt_cfl

        domain_width = self.rho_max - self.rho_min
        if inner_sponge_width is None:
            inner_sponge_width = min(0.35 * self.mass, 0.12 * domain_width)
        if outer_sponge_width is None:
            outer_sponge_width = min(2.0 * self.mass, 0.20 * domain_width)
        self.inner_sponge_width = float(inner_sponge_width)
        self.outer_sponge_width = float(outer_sponge_width)
        self._validate_sponges()
        self.physical_rho_min = self.rho_min + self.inner_sponge_width
        self.physical_rho_max = self.rho_max - self.outer_sponge_width

        self.Hz = np.zeros((self.Nr, self.Nphi), dtype=self.dtype)
        self.Er = np.zeros((self.Nr, self.Nphi), dtype=self.dtype)
        self.Ephi = np.zeros((self.Nr + 1, self.Nphi), dtype=self.dtype)

        sigma_h = self._sponge_sigma(self.rho_centers)
        sigma_er = sigma_h.copy()
        sigma_ephi = self._sponge_sigma(self.rho_faces)
        self._damp_h_half = np.exp(-0.5 * self.dt * sigma_h)[:, None]
        self._damp_er_half = np.exp(-0.5 * self.dt * sigma_er)[:, None]
        self._damp_ephi_half = np.exp(-0.5 * self.dt * sigma_ephi)[:, None]

        self.time = 0.0
        self.step_count = 0
        self.packet: Optional[Dict[str, float]] = None
        self.history: Optional[Dict[str, np.ndarray | list]] = None
        self.backend_requested = "python"
        self.backend = "python"
        self._cython_kernel = _cython_kernel_gr
        self._gpu_state = None
        self._gpu_host_dirty = False
        self._gpu_transfer_stats = {
            "host_to_device": 0,
            "device_to_host": 0,
            "host_to_device_during_steps": 0,
            "device_to_host_during_steps": 0,
        }

    @property
    def horizon_radius(self) -> float:
        return self.geometry.horizon_isotropic_radius

    @property
    def photon_sphere_radius(self) -> float:
        return self.geometry.photon_sphere_isotropic_radius

    @property
    def photon_orbit_period(self) -> float:
        return self.geometry.photon_orbit_period

    def config(self, backend: str = "cpu"):
        """Select ``cpu`` (Cython), ``gpu`` (Numba-CUDA), or ``python``.

        Missing optional runtimes fall back to the NumPy reference update with
        a warning.  The CUDA path keeps mutable fields on the device across
        calls and transfers them back only for a requested diagnostic, history
        sample, :meth:`sync_fields`, or the end of :meth:`run`.
        """

        requested = str(backend).lower().replace("-", "_")
        if requested not in {"cpu", "gpu", "python"}:
            raise ValueError("backend must be 'cpu', 'gpu', or 'python'.")
        self._discard_gpu_state(preserve_fields=True)
        self.backend_requested = requested

        if requested == "cpu":
            compatible = self.dtype == np.dtype(np.complex128)
            if self._cython_kernel is not None and compatible:
                self.backend = "cython"
            else:
                self.backend = "python"
                reason = (
                    "the compiled extension is unavailable"
                    if self._cython_kernel is None
                    else "the compiled kernel currently requires complex128 fields"
                )
                warnings.warn(
                    f"GR Cython backend requested but {reason}; using NumPy.",
                    RuntimeWarning,
                )
        elif requested == "gpu":
            available = False
            try:
                from numba import cuda

                available = bool(cuda.is_available())
                if available:
                    from . import cuda_gr  # noqa: F401
            except Exception:
                available = False
            if available:
                self.backend = "numba_cuda"
            else:
                self.backend = "python"
                warnings.warn(
                    "Numba-CUDA is unavailable; using the NumPy GR update.",
                    RuntimeWarning,
                )
        else:
            self.backend = "python"
        return self

    def _discard_gpu_state(self, preserve_fields: bool) -> None:
        if getattr(self, "_gpu_state", None) is None:
            return
        from . import cuda_gr

        if preserve_fields:
            cuda_gr.sync_to_host(self)
        cuda_gr.discard_state(self)

    def _ensure_host_current(self) -> None:
        if getattr(self, "_gpu_state", None) is None:
            return
        from . import cuda_gr

        cuda_gr.sync_to_host(self)

    def sync_fields(self):
        """Synchronize device-resident fields to their public NumPy arrays."""

        self._ensure_host_current()
        return self

    def _validate_sponges(self) -> None:
        for name, width in (
            ("inner_sponge_width", self.inner_sponge_width),
            ("outer_sponge_width", self.outer_sponge_width),
        ):
            if not np.isfinite(width) or width < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.inner_sponge_width + self.outer_sponge_width >= (
            self.rho_max - self.rho_min
        ):
            raise ValueError("The inner and outer sponge layers must not overlap.")

    def _sponge_sigma(self, rho: np.ndarray) -> np.ndarray:
        sigma = np.zeros_like(rho, dtype=float)
        log_attenuation = -np.log(self.sponge_reflection)

        if self.inner_sponge_width > 0.0:
            inner_edge = self.rho_min + self.inner_sponge_width
            q_inner = np.clip(
                (inner_edge - rho) / self.inner_sponge_width, 0.0, 1.0
            )
            sigma_inner_max = (
                log_attenuation * (self.sponge_order + 1)
                / self.inner_sponge_width
            )
            sigma += sigma_inner_max * q_inner**self.sponge_order

        if self.outer_sponge_width > 0.0:
            outer_edge = self.rho_max - self.outer_sponge_width
            q_outer = np.clip(
                (rho - outer_edge) / self.outer_sponge_width, 0.0, 1.0
            )
            sigma_outer_max = (
                log_attenuation * (self.sponge_order + 1)
                / self.outer_sponge_width
            )
            sigma += sigma_outer_max * q_outer**self.sponge_order

        return sigma

    @staticmethod
    def _wrapped_angle(phi: np.ndarray, origin: float) -> np.ndarray:
        return (phi - origin + np.pi) % (2.0 * np.pi) - np.pi

    def clear_fields(self) -> None:
        self._discard_gpu_state(preserve_fields=False)
        self.Hz.fill(0.0)
        self.Er.fill(0.0)
        self.Ephi.fill(0.0)
        self.time = 0.0
        self.step_count = 0
        self.packet = None
        self.history = None

    def initialize_orbiting_packet(
        self,
        phi0: float = 0.0,
        direction: int = 1,
        azimuthal_mode: int = 16,
        radial_width: Optional[float] = None,
        angular_width: float = 0.28,
        amplitude: float = 1.0,
        rho0: Optional[float] = None,
    ) -> Dict[str, float]:
        """Place a divergence-free, nearly tangential wave packet on the grid.

        By default the packet is centred on the Schwarzschild photon sphere.
        ``direction=+1`` moves toward increasing azimuth and ``-1`` reverses it.
        The carrier uses an integer azimuthal mode, while the Gaussian envelope
        localizes it to a small arc.  A finite packet will spread and leak from
        the unstable photon orbit; that behavior is physical.
        """

        if direction not in {-1, 1}:
            raise ValueError("direction must be +1 or -1.")
        if not isinstance(azimuthal_mode, (int, np.integer)) or azimuthal_mode < 2:
            raise ValueError("azimuthal_mode must be an integer of at least 2.")
        if not np.isfinite(angular_width) or not 0.0 < angular_width < np.pi:
            raise ValueError("angular_width must lie between zero and pi.")
        if not np.isfinite(amplitude) or amplitude == 0.0:
            raise ValueError("amplitude must be finite and non-zero.")

        if rho0 is None:
            rho0 = self.photon_sphere_radius
        rho0 = float(rho0)
        if not self.rho_min < rho0 < self.rho_max:
            raise ValueError("rho0 must lie strictly inside the radial domain.")
        if radial_width is None:
            radial_width = 0.35 * self.mass
        radial_width = float(radial_width)
        if not np.isfinite(radial_width) or radial_width <= 0.0:
            raise ValueError("radial_width must be finite and positive.")
        if rho0 - 3.0 * radial_width <= self.rho_min:
            raise ValueError("The packet overlaps the inner boundary; reduce radial_width.")
        if rho0 + 3.0 * radial_width >= self.rho_max:
            raise ValueError("The packet overlaps the outer boundary; reduce radial_width.")

        self.clear_fields()
        phi0 = float(phi0) % (2.0 * np.pi)
        mode = int(azimuthal_mode)
        delta_phi = self._wrapped_angle(self.phi_centers, phi0)
        radial_envelope = np.exp(
            -0.5 * ((self.rho_centers - rho0) / radial_width) ** 2
        )
        angular_envelope = np.exp(-0.5 * (delta_phi / angular_width) ** 2)
        carrier = np.exp(1j * direction * mode * delta_phi)
        hz_at_t0 = (
            float(amplitude)
            * radial_envelope[:, None]
            * angular_envelope[None, :]
            * carrier[None, :]
        ).astype(self.dtype, copy=False)

        n0 = float(self.geometry.refractive_index(rho0))
        angular_velocity = direction / (n0 * rho0)
        omega = abs(mode * angular_velocity)

        # Construct D as a discrete rotated gradient of Hz.  The compatible
        # radial/azimuthal differences make div(D)=0 to roundoff away from the
        # radial boundaries, avoiding a spurious electrostatic launch mode.
        dr_field = (
            1j
            / omega
            * (hz_at_t0 - np.roll(hz_at_t0, 1, axis=1))
            / (self.rho_centers[:, None] * self.dphi)
        )
        dphi_field = np.zeros_like(self.Ephi)
        dphi_field[1:-1, :] = (
            -1j
            / omega
            * (hz_at_t0[1:, :] - hz_at_t0[:-1, :])
            / self.drho
        )
        self.Er[...] = dr_field / self.n_er[:, None]
        self.Ephi[...] = dphi_field / self.n_ephi[:, None]

        # E is stored at t=0 and H at t=-dt/2 for the leapfrog update.
        self.Hz[...] = hz_at_t0 * np.exp(0.5j * omega * self.dt)
        self._apply_radial_boundary()
        self.packet = {
            "rho0": rho0,
            "phi0": phi0,
            "direction": float(direction),
            "azimuthal_mode": float(mode),
            "radial_width": radial_width,
            "angular_width": float(angular_width),
            "omega": omega,
            "angular_velocity": angular_velocity,
            "coordinate_period": 2.0 * np.pi / abs(angular_velocity),
        }
        return dict(self.packet)

    def _apply_radial_boundary(self) -> None:
        if self.radial_boundary == "pec":
            self.Ephi[0, :] = 0.0
            self.Ephi[-1, :] = 0.0
            return

        # Unit impedance follows from eps_r=mu_r.  At the inner boundary the
        # accepted characteristic travels toward decreasing rho; at the outer
        # boundary it travels toward increasing rho.
        self.Ephi[0, :] = -self.Hz[0, :]
        self.Ephi[-1, :] = self.Hz[-1, :]

    def step(self, count: int = 1) -> None:
        """Advance ``count`` leapfrog steps with the selected backend."""

        if not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError("count must be a positive integer.")
        count = int(count)

        if self.backend == "cython":
            self._cython_kernel.step_fields(
                self.Hz,
                self.Er,
                self.Ephi,
                self.rho_centers,
                self.rho_faces,
                self.n_hz,
                self.n_er,
                self.n_ephi,
                self._damp_h_half[:, 0],
                self._damp_er_half[:, 0],
                self._damp_ephi_half[:, 0],
                self.dt,
                self.drho,
                self.dphi,
                1 if self.radial_boundary == "pec" else 0,
                count,
            )
            self.time += count * self.dt
            self.step_count += count
            return

        if self.backend == "numba_cuda":
            from . import cuda_gr

            cuda_gr.run_steps(self, count)
            self.time += count * self.dt
            self.step_count += count
            return

        self._step_numpy(count)

    def _step_numpy(self, count: int) -> None:
        """Reference NumPy implementation of the polar GR Yee update."""

        radial_metric = self.rho_centers[:, None]
        for _ in range(count):
            # Strang-split matched electric/magnetic attenuation.
            self.Er *= self._damp_er_half
            self.Ephi *= self._damp_ephi_half
            self._apply_radial_boundary()

            radial_curl = (
                self.rho_faces[1:, None] * self.Ephi[1:, :]
                - self.rho_faces[:-1, None] * self.Ephi[:-1, :]
            ) / self.drho
            angular_curl = (
                np.roll(self.Er, -1, axis=1) - self.Er
            ) / self.dphi
            curl_e = (radial_curl - angular_curl) / radial_metric

            self.Hz *= self._damp_h_half
            self.Hz -= self.dt * curl_e / self.n_hz[:, None]
            self.Hz *= self._damp_h_half

            d_hz_dphi = (
                self.Hz - np.roll(self.Hz, 1, axis=1)
            ) / (radial_metric * self.dphi)
            self.Er += self.dt * d_hz_dphi / self.n_er[:, None]

            d_hz_drho = (self.Hz[1:, :] - self.Hz[:-1, :]) / self.drho
            self.Ephi[1:-1, :] -= (
                self.dt * d_hz_drho / self.n_ephi[1:-1, None]
            )
            self._apply_radial_boundary()
            self.Er *= self._damp_er_half
            self.Ephi *= self._damp_ephi_half
            self._apply_radial_boundary()

            self.time += self.dt
            self.step_count += 1

    def fields_at_centers(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``Er, Ephi, Hz`` interpolated to common cell centres."""

        self._ensure_host_current()
        er_center = 0.5 * (self.Er + np.roll(self.Er, -1, axis=1))
        ephi_center = 0.5 * (self.Ephi[:-1, :] + self.Ephi[1:, :])
        return er_center, ephi_center, self.Hz

    def energy_density(self) -> np.ndarray:
        """Return positive normalized analytic-signal energy at cell centres."""

        er_center, ephi_center, hz_center = self.fields_at_centers()
        return 0.5 * self.n_hz[:, None] * (
            np.abs(er_center) ** 2
            + np.abs(ephi_center) ** 2
            + np.abs(hz_center) ** 2
        )

    def total_energy(self) -> float:
        density = self.energy_density()
        area = self.rho_centers[:, None] * self.drho * self.dphi
        return float(np.sum(density * area))

    def conserved_energy(self) -> float:
        """Return the exact lossless leapfrog invariant for a closed annulus.

        This diagnostic is defined for ``radial_boundary='pec'`` with both
        sponge widths zero.  It pairs the current half-step magnetic field with
        the next half-step value predicted from the current electric field.
        Unlike :meth:`total_energy`, it should remain constant to roundoff for
        the semi-discrete lossless update (subject to floating-point error).
        """

        self._ensure_host_current()
        if self.radial_boundary != "pec":
            raise RuntimeError("conserved_energy requires radial_boundary='pec'.")
        if self.inner_sponge_width != 0.0 or self.outer_sponge_width != 0.0:
            raise RuntimeError("conserved_energy requires both sponges disabled.")

        radial_metric = self.rho_centers[:, None]
        radial_curl = (
            self.rho_faces[1:, None] * self.Ephi[1:, :]
            - self.rho_faces[:-1, None] * self.Ephi[:-1, :]
        ) / self.drho
        angular_curl = (
            np.roll(self.Er, -1, axis=1) - self.Er
        ) / self.dphi
        curl_e = (radial_curl - angular_curl) / radial_metric
        hz_next = self.Hz - self.dt * curl_e / self.n_hz[:, None]

        edge_measure_r = self.rho_centers[:, None] * self.drho * self.dphi
        edge_measure_phi = self.rho_faces[:, None] * self.drho * self.dphi
        cell_measure = edge_measure_r
        electric = np.sum(
            self.n_er[:, None] * np.abs(self.Er) ** 2 * edge_measure_r
        )
        electric += np.sum(
            self.n_ephi[:, None] * np.abs(self.Ephi) ** 2 * edge_measure_phi
        )
        magnetic = np.real(
            np.sum(
                self.n_hz[:, None]
                * np.conj(self.Hz)
                * hz_next
                * cell_measure
            )
        )
        return float(0.5 * (electric + magnetic))

    def poynting(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return normalized cycle-averaged ``(S_rho, S_phi)`` at centres."""

        er_center, ephi_center, hz_center = self.fields_at_centers()
        s_rho = 0.5 * np.real(ephi_center * np.conj(hz_center))
        s_phi = -0.5 * np.real(er_center * np.conj(hz_center))
        return s_rho, s_phi

    def electric_divergence(self, active_only: bool = False) -> np.ndarray:
        """Return discrete ``div(D)`` at interior dual-grid vertices.

        ``active_only=True`` excludes the two matched sponges.  Spatially
        varying attenuation represents effective charge inside those layers,
        while the undamped Yee update preserves the source-free constraint.
        """

        self._ensure_host_current()
        d_r = self.n_er[:, None] * self.Er
        d_phi = self.n_ephi[:, None] * self.Ephi
        radius = self.rho_faces[1:-1, None]
        radial_part = (
            self.rho_centers[1:, None] * d_r[1:, :]
            - self.rho_centers[:-1, None] * d_r[:-1, :]
        ) / (radius * self.drho)
        angular_part = (
            d_phi[1:-1, :] - np.roll(d_phi[1:-1, :], 1, axis=1)
        ) / (radius * self.dphi)
        divergence = radial_part + angular_part
        if not active_only:
            return divergence
        vertex_radii = self.rho_faces[1:-1]
        active = (
            (vertex_radii >= self.physical_rho_min + self.drho)
            & (vertex_radii <= self.physical_rho_max - self.drho)
        )
        return divergence[active, :]

    def diagnostics(self) -> Dict[str, float]:
        density = self.energy_density()
        weights = density * self.rho_centers[:, None]
        weight_sum = float(np.sum(weights))
        if weight_sum <= np.finfo(float).tiny:
            return {
                "time": self.time,
                "energy": 0.0,
                "rho_mean": np.nan,
                "phi_mean": np.nan,
                "phi_coherence": 0.0,
                "rho_peak": np.nan,
                "phi_peak": np.nan,
                "divergence_linf": 0.0,
                "divergence_linf_global": 0.0,
            }

        rho_mean = float(
            np.sum(weights * self.rho_centers[:, None]) / weight_sum
        )
        circular_moment = np.sum(
            weights * np.exp(1j * self.phi_centers[None, :])
        )
        phi_mean = float(np.angle(circular_moment) % (2.0 * np.pi))
        phi_coherence = min(1.0, float(np.abs(circular_moment) / weight_sum))
        if np.isclose(phi_mean, 2.0 * np.pi, rtol=0.0, atol=1.0e-12):
            phi_mean = 0.0
        peak_i, peak_j = np.unravel_index(np.argmax(density), density.shape)
        divergence = self.electric_divergence(active_only=True)
        divergence_global = self.electric_divergence(active_only=False)
        return {
            "time": self.time,
            "energy": self.total_energy(),
            "rho_mean": rho_mean,
            "phi_mean": phi_mean,
            "phi_coherence": phi_coherence,
            "rho_peak": float(self.rho_centers[peak_i]),
            "phi_peak": float(self.phi_centers[peak_j]),
            "divergence_linf": (
                float(np.max(np.abs(divergence))) if divergence.size else 0.0
            ),
            "divergence_linf_global": float(np.max(np.abs(divergence_global))),
        }

    def _record_state(
        self,
        records: Dict[str, list],
        store_snapshots: bool,
        snapshot_quantity: str,
    ) -> None:
        sample = self.diagnostics()
        for key in (
            "time",
            "energy",
            "rho_mean",
            "phi_mean",
            "phi_coherence",
            "rho_peak",
            "phi_peak",
            "divergence_linf",
            "divergence_linf_global",
        ):
            records[key].append(sample[key])

        if store_snapshots:
            if snapshot_quantity == "energy":
                snapshot = self.energy_density()
            elif snapshot_quantity == "real_hz":
                snapshot = np.real(self.Hz)
            elif snapshot_quantity == "abs_hz":
                snapshot = np.abs(self.Hz)
            else:
                raise ValueError(
                    "snapshot_quantity must be 'energy', 'real_hz', or 'abs_hz'."
                )
            records["snapshots"].append(np.asarray(snapshot, dtype=np.float32))

    def run(
        self,
        *,
        steps: Optional[int] = None,
        duration: Optional[float] = None,
        record_stride: int = 10,
        store_snapshots: bool = False,
        snapshot_quantity: str = "energy",
        progress: bool = False,
    ) -> Dict[str, np.ndarray | list]:
        """Run the solver and return centroid, energy, and optional field history."""

        if (steps is None) == (duration is None):
            raise ValueError("Specify exactly one of steps or duration.")
        if duration is not None:
            if not np.isfinite(duration) or duration <= 0.0:
                raise ValueError("duration must be finite and positive.")
            steps = int(np.ceil(float(duration) / self.dt))
        if not isinstance(steps, (int, np.integer)) or steps < 1:
            raise ValueError("steps must be a positive integer.")
        if not isinstance(record_stride, (int, np.integer)) or record_stride < 1:
            raise ValueError("record_stride must be a positive integer.")
        maximum_sample_angle = (
            int(record_stride)
            * self.dt
            * self.geometry.photon_orbit_angular_velocity
        )
        if maximum_sample_angle >= np.pi:
            warnings.warn(
                "record_stride permits more than pi radians of angular motion "
                "between diagnostics; phi_unwrapped may alias.",
                RuntimeWarning,
            )

        records: Dict[str, list] = {
            "time": [],
            "energy": [],
            "rho_mean": [],
            "phi_mean": [],
            "phi_coherence": [],
            "rho_peak": [],
            "phi_peak": [],
            "divergence_linf": [],
            "divergence_linf_global": [],
            "snapshots": [],
        }
        self._record_state(records, store_snapshots, snapshot_quantity)

        progress_bar = None
        if progress:
            try:
                from tqdm import tqdm

                progress_bar = tqdm(
                    total=int(steps), desc="Schwarzschild FDTD", unit="step"
                )
            except ImportError:
                pass

        completed = 0
        try:
            while completed < int(steps):
                chunk = min(int(record_stride), int(steps) - completed)
                self.step(chunk)
                completed += chunk
                self._record_state(records, store_snapshots, snapshot_quantity)
                if progress_bar is not None:
                    progress_bar.update(chunk)
        finally:
            if progress_bar is not None:
                progress_bar.close()

        phi_raw = np.asarray(records["phi_mean"], dtype=float)
        phi_unwrapped = np.unwrap(phi_raw)
        history: Dict[str, np.ndarray | list] = {
            key: np.asarray(value, dtype=float)
            for key, value in records.items()
            if key != "snapshots"
        }
        history["phi_unwrapped"] = phi_unwrapped
        history["snapshots"] = records["snapshots"]
        history["snapshot_quantity"] = snapshot_quantity
        self.history = history
        return history

    def expected_packet_angle(self, times: np.ndarray | float) -> np.ndarray:
        """Return the launch's ideal angular phase reference.

        At the photon sphere this is the circular Schwarzschild null ray.  For
        a custom ``rho0`` it is only the local tangential phase-speed reference,
        because no other constant-radius Schwarzschild null geodesic exists.
        """

        if self.packet is None:
            raise RuntimeError("initialize_orbiting_packet must be called first.")
        return self.packet["phi0"] + self.packet["angular_velocity"] * np.asarray(times)

    def plot_snapshot(
        self,
        data: Optional[np.ndarray] = None,
        *,
        quantity: str = "energy",
        ax=None,
        view_radius: Optional[float] = None,
        log_scale: bool = False,
        cmap: Optional[str] = None,
        title: Optional[str] = None,
        colorbar: bool = True,
    ):
        """Plot a Cartesian view of the polar field grid."""

        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        from matplotlib.patches import Circle

        if data is None:
            self._ensure_host_current()
            if quantity == "energy":
                data = self.energy_density()
            elif quantity == "real_hz":
                data = np.real(self.Hz)
            elif quantity == "abs_hz":
                data = np.abs(self.Hz)
            else:
                raise ValueError("quantity must be 'energy', 'real_hz', or 'abs_hz'.")
        values = np.asarray(data)
        if values.shape != (self.Nr, self.Nphi):
            raise ValueError(
                f"snapshot data must have shape {(self.Nr, self.Nphi)}, "
                f"not {values.shape}."
            )
        if np.iscomplexobj(values):
            raise ValueError("snapshot data must be real-valued.")

        if ax is None:
            _, ax = plt.subplots(figsize=(7.2, 7.0))
        phi_edges = np.linspace(0.0, 2.0 * np.pi, self.Nphi + 1)
        x_edges = (
            self.rho_faces[:, None] * np.cos(phi_edges[None, :]) / self.mass
        )
        y_edges = (
            self.rho_faces[:, None] * np.sin(phi_edges[None, :]) / self.mass
        )

        if cmap is None:
            cmap = "magma" if quantity != "real_hz" else "RdBu_r"
        norm = None
        if log_scale:
            positive = values[values > 0.0]
            if positive.size:
                vmax = float(np.max(positive))
                vmin = max(float(np.percentile(positive, 2.0)), vmax * 1.0e-8)
                norm = LogNorm(vmin=vmin, vmax=vmax)
        mesh = ax.pcolormesh(
            x_edges, y_edges, values, shading="flat", cmap=cmap, norm=norm
        )
        # The isotropic chart stops outside the horizon.  Fill the complete
        # excised disk black, then mark the true horizon within it.
        ax.add_patch(
            Circle(
                (0.0, 0.0),
                self.rho_min / self.mass,
                color="black",
                zorder=5,
            )
        )
        ax.add_patch(
            Circle(
                (0.0, 0.0),
                self.horizon_radius / self.mass,
                fill=False,
                linewidth=1.0,
                edgecolor="white",
                zorder=6,
            )
        )
        ax.add_patch(
            Circle(
                (0.0, 0.0),
                self.photon_sphere_radius / self.mass,
                fill=False,
                linestyle="--",
                linewidth=1.2,
                color="cyan",
                alpha=0.9,
                zorder=6,
            )
        )
        if view_radius is None:
            view_radius = min(self.rho_max, 4.0 * self.mass)
        ax.set_xlim(-view_radius / self.mass, view_radius / self.mass)
        ax.set_ylim(-view_radius / self.mass, view_radius / self.mass)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x/M$")
        ax.set_ylabel(r"$y/M$")
        if title is None:
            title = f"{quantity.replace('_', ' ')} at t/M = {self.time / self.mass:.3f}"
        ax.set_title(title)
        if colorbar:
            ax.figure.colorbar(mesh, ax=ax, shrink=0.82, label=quantity.replace("_", " "))
        return ax, mesh

    def plot_diagnostics(self, history: Optional[Dict[str, np.ndarray | list]] = None):
        """Plot packet radius, unwrapped angle, and retained field energy."""

        import matplotlib.pyplot as plt

        if history is None:
            history = self.history
        if history is None:
            raise RuntimeError("run the simulation before plotting diagnostics.")

        time = np.asarray(history["time"], dtype=float)
        radius = np.asarray(history["rho_mean"], dtype=float)
        angle = np.asarray(history["phi_unwrapped"], dtype=float)
        energy = np.asarray(history["energy"], dtype=float)
        coherence = np.asarray(history.get("phi_coherence", []), dtype=float)

        fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.5), sharex=True)
        axes[0].plot(time / self.mass, radius / self.mass, label="wave centroid")
        axes[0].axhline(
            self.photon_sphere_radius / self.mass,
            color="black",
            linestyle="--",
            label="photon sphere",
        )
        axes[0].set_ylabel(r"$\langle\rho\rangle/M$")
        axes[0].legend(loc="best")

        axes[1].plot(time / self.mass, angle, label="wave centroid")
        if self.packet is not None:
            expected_angle = self.expected_packet_angle(time)
            expected_angle += angle[0] - expected_angle[0]
            is_photon_orbit = np.isclose(
                self.packet["rho0"],
                self.photon_sphere_radius,
                rtol=1.0e-12,
                atol=1.0e-12 * self.mass,
            )
            axes[1].plot(
                time / self.mass,
                expected_angle,
                linestyle="--",
                label=(
                    "circular null ray"
                    if is_photon_orbit
                    else "local tangential phase reference"
                ),
            )
        axes[1].set_ylabel(r"unwrapped $\phi$")
        axes[1].legend(loc="best")

        baseline = energy[0] if energy.size and energy[0] != 0.0 else 1.0
        axes[2].plot(time / self.mass, energy / baseline, label="energy / initial")
        if coherence.size == time.size:
            axes[2].plot(
                time / self.mass,
                coherence,
                linestyle=":",
                label=r"angular coherence $|\langle e^{i\phi}\rangle|$",
            )
            axes[2].legend(loc="best")
        axes[2].set_ylabel("normalized value")
        axes[2].set_xlabel(r"coordinate time $t/M$")
        fig.tight_layout()
        return fig, axes

    def save_animation(
        self,
        path: str | Path,
        history: Optional[Dict[str, np.ndarray | list]] = None,
        *,
        fps: int = 24,
        view_radius: Optional[float] = None,
    ) -> Path:
        """Save recorded snapshots, using FFmpeg for MP4 output."""

        import matplotlib.pyplot as plt
        from matplotlib.animation import (
            FFMpegWriter,
            FuncAnimation,
            PillowWriter,
            writers,
        )
        from matplotlib.patches import Circle

        if not isinstance(fps, (int, np.integer)) or fps < 1:
            raise ValueError("fps must be a positive integer.")
        if history is None:
            history = self.history
        if history is None or not history.get("snapshots"):
            raise RuntimeError("run with store_snapshots=True before animation.")
        snapshots = history["snapshots"]
        times = np.asarray(history["time"], dtype=float)
        quantity = str(history.get("snapshot_quantity", "energy"))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        if suffix == ".mp4" and not writers.is_available("ffmpeg"):
            raise RuntimeError(
                "Saving an MP4 animation requires FFmpeg on the executable path."
            )

        phi_edges = np.linspace(0.0, 2.0 * np.pi, self.Nphi + 1)
        x_edges = (
            self.rho_faces[:, None] * np.cos(phi_edges[None, :]) / self.mass
        )
        y_edges = (
            self.rho_faces[:, None] * np.sin(phi_edges[None, :]) / self.mass
        )
        if view_radius is None:
            view_radius = min(self.rho_max, 4.0 * self.mass)

        fig, ax = plt.subplots(figsize=(7.2, 7.0))
        first = np.asarray(snapshots[0])
        vmax = max(float(np.max(np.abs(frame))) for frame in snapshots)
        if quantity == "real_hz":
            mesh = ax.pcolormesh(
                x_edges,
                y_edges,
                first,
                shading="flat",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
            )
        else:
            mesh = ax.pcolormesh(
                x_edges,
                y_edges,
                first,
                shading="flat",
                cmap="magma",
                vmin=0.0,
                vmax=vmax,
            )
        ax.add_patch(
            Circle((0.0, 0.0), self.rho_min / self.mass, color="black")
        )
        ax.add_patch(
            Circle(
                (0.0, 0.0),
                self.horizon_radius / self.mass,
                fill=False,
                linewidth=1.0,
                edgecolor="white",
            )
        )
        ax.add_patch(
            Circle(
                (0.0, 0.0),
                self.photon_sphere_radius / self.mass,
                fill=False,
                linestyle="--",
                linewidth=1.2,
                color="cyan",
            )
        )
        ax.set_xlim(-view_radius / self.mass, view_radius / self.mass)
        ax.set_ylim(-view_radius / self.mass, view_radius / self.mass)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x/M$")
        ax.set_ylabel(r"$y/M$")
        title = ax.set_title("")
        fig.colorbar(mesh, ax=ax, shrink=0.82, label=quantity.replace("_", " "))

        def update(frame_index: int):
            mesh.set_array(np.asarray(snapshots[frame_index]).ravel())
            title.set_text(f"t/M = {times[frame_index] / self.mass:.3f}")
            return mesh, title

        animation = FuncAnimation(
            fig, update, frames=len(snapshots), interval=1000.0 / fps, blit=False
        )
        if suffix == ".gif":
            animation.save(path, writer=PillowWriter(fps=fps))
        elif suffix == ".mp4":
            animation.save(
                path,
                writer=FFMpegWriter(
                    fps=fps,
                    codec="h264",
                    extra_args=["-pix_fmt", "yuv420p"],
                ),
            )
        else:
            animation.save(path, fps=fps)
        plt.close(fig)
        return path

    def to_physical_length(self, coordinate_length: np.ndarray | float, mass_solar: float):
        """Convert solver length to metres for a black hole mass in solar masses."""

        if not np.isfinite(mass_solar) or mass_solar <= 0.0:
            raise ValueError("mass_solar must be finite and positive.")
        physical_mass = mass_solar * self.SOLAR_MASS_KG
        one_m_in_metres = self.G_SI * physical_mass / self.C_SI**2
        return np.asarray(coordinate_length) / self.mass * one_m_in_metres

    def to_physical_time(self, coordinate_time: np.ndarray | float, mass_solar: float):
        """Convert solver time to seconds for a black hole mass in solar masses."""

        if not np.isfinite(mass_solar) or mass_solar <= 0.0:
            raise ValueError("mass_solar must be finite and positive.")
        physical_mass = mass_solar * self.SOLAR_MASS_KG
        one_m_in_seconds = self.G_SI * physical_mass / self.C_SI**3
        return np.asarray(coordinate_time) / self.mass * one_m_in_seconds


__all__ = ["FDTD_2D_GR", "SchwarzschildGeometry"]
