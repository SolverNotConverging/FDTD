"""Device-resident Numba-CUDA stepping for :mod:`FDTD_2D_GR`.

The public functions in this module deliberately operate on a solver instance
instead of defining a second solver class.  The first call to :func:`run_steps`
uploads the three fields and the immutable metric/damping arrays.  Subsequent
calls reuse that state, and :func:`sync_to_host` is the only normal path that
copies fields back to NumPy.

The kernel order mirrors ``FDTD_2D_GR.step``.  In particular, radial boundary
values are imposed after each electric half-damping operation, azimuth is
periodic, and the characteristic boundary uses the current half-step magnetic
field.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from numba import cuda


THREADS_2D = (16, 16)
_STATE_ATTRIBUTE = "_gr_cuda_state"


@cuda.jit
def _damp_electric_and_apply_boundary(
    er,
    ephi,
    hz,
    damp_er_half,
    damp_ephi_half,
    characteristic,
):
    """Apply one electric half-loss step and the radial boundary values."""

    i, j = cuda.grid(2)
    nr = hz.shape[0]
    nphi = hz.shape[1]
    if j >= nphi:
        return

    if i < nr:
        er[i, j] *= damp_er_half[i, 0]
    if i <= nr:
        ephi[i, j] *= damp_ephi_half[i, 0]
        if i == 0:
            ephi[i, j] = -hz[0, j] if characteristic else 0.0
        elif i == nr:
            ephi[i, j] = hz[nr - 1, j] if characteristic else 0.0


@cuda.jit
def _update_magnetic(
    hz,
    er,
    ephi,
    rho_faces,
    rho_centers,
    n_hz,
    damp_h_half,
    drho,
    dphi,
    dt,
):
    """Update ``Hz`` with the polar electric curl and matched loss."""

    i, j = cuda.grid(2)
    nr, nphi = hz.shape
    if i >= nr or j >= nphi:
        return

    j_next = j + 1
    if j_next == nphi:
        j_next = 0
    radial_curl = (
        rho_faces[i + 1] * ephi[i + 1, j]
        - rho_faces[i] * ephi[i, j]
    ) / drho
    angular_curl = (er[i, j_next] - er[i, j]) / dphi
    curl_e = (radial_curl - angular_curl) / rho_centers[i]

    value = hz[i, j] * damp_h_half[i, 0]
    value -= dt * curl_e / n_hz[i]
    hz[i, j] = value * damp_h_half[i, 0]


@cuda.jit
def _update_electric(
    er,
    ephi,
    hz,
    rho_centers,
    n_er,
    n_ephi,
    drho,
    dphi,
    dt,
):
    """Update both staggered electric components from the new ``Hz``."""

    i, j = cuda.grid(2)
    nr, nphi = hz.shape
    if j >= nphi:
        return

    if i < nr:
        j_previous = j - 1
        if j_previous < 0:
            j_previous = nphi - 1
        d_hz_dphi = (
            hz[i, j] - hz[i, j_previous]
        ) / (rho_centers[i] * dphi)
        er[i, j] += dt * d_hz_dphi / n_er[i]

    if 0 < i < nr:
        d_hz_drho = (hz[i, j] - hz[i - 1, j]) / drho
        ephi[i, j] -= dt * d_hz_drho / n_ephi[i]


def is_available() -> bool:
    """Return whether Numba reports a usable CUDA runtime.

    This also returns ``True`` when ``NUMBA_ENABLE_CUDASIM=1`` is active.
    Driver-discovery failures are treated as an unavailable runtime.
    """

    try:
        return bool(cuda.is_available())
    except Exception:
        return False


cuda_available = is_available


def availability() -> Dict[str, Any]:
    """Return a small, side-effect-free CUDA availability description."""

    simulated = bool(getattr(cuda.config, "ENABLE_CUDASIM", False))
    return {
        "available": is_available(),
        "simulator": simulated,
        "runtime": "numba_cuda",
    }


def _blocks_2d(sim) -> tuple[int, int]:
    return (
        (int(sim.Nr) + 1 + THREADS_2D[0] - 1) // THREADS_2D[0],
        (int(sim.Nphi) + THREADS_2D[1] - 1) // THREADS_2D[1],
    )


def _device_copy(array: np.ndarray):
    return cuda.to_device(np.ascontiguousarray(array))


def _get_state(sim):
    state = getattr(sim, _STATE_ATTRIBUTE, None)
    if state is None:
        candidate = getattr(sim, "_gpu_state", None)
        if isinstance(candidate, dict) and candidate.get("runtime") == "gr_cuda":
            state = candidate
    return state


def _new_state(sim):
    if not is_available():
        raise RuntimeError(
            "The Numba-CUDA runtime is unavailable. Install a compatible CUDA "
            "driver or set NUMBA_ENABLE_CUDASIM=1 for simulator testing."
        )

    required = (
        "Hz",
        "Er",
        "Ephi",
        "rho_faces",
        "rho_centers",
        "n_hz",
        "n_er",
        "n_ephi",
        "_damp_h_half",
        "_damp_er_half",
        "_damp_ephi_half",
    )
    missing = [name for name in required if not hasattr(sim, name)]
    if missing:
        raise AttributeError(
            "FDTD_2D_GR instance is missing CUDA state arrays: "
            + ", ".join(missing)
        )

    state = {
        "runtime": "gr_cuda",
        "Hz": _device_copy(sim.Hz),
        "Er": _device_copy(sim.Er),
        "Ephi": _device_copy(sim.Ephi),
        "rho_faces": _device_copy(sim.rho_faces),
        "rho_centers": _device_copy(sim.rho_centers),
        "n_hz": _device_copy(sim.n_hz),
        "n_er": _device_copy(sim.n_er),
        "n_ephi": _device_copy(sim.n_ephi),
        "damp_h_half": _device_copy(sim._damp_h_half),
        "damp_er_half": _device_copy(sim._damp_er_half),
        "damp_ephi_half": _device_copy(sim._damp_ephi_half),
        "host_dirty": False,
    }
    setattr(sim, _STATE_ATTRIBUTE, state)
    # Keep the conventional repository attribute as an alias so generic state
    # serialization/debugging code can recognize device-resident state.
    sim._gpu_state = state
    sim._gpu_host_dirty = False
    sim._gpu_transfer_stats = {
        "host_to_device": 11,
        "device_to_host": 0,
        "host_to_device_initial": 11,
        "device_to_host_sync": 0,
        "host_to_device_during_steps": 0,
        "device_to_host_during_steps": 0,
        "state_uploads": 1,
        "host_syncs": 0,
        "steps": 0,
    }
    return state


def _ensure_state(sim):
    state = _get_state(sim)
    if state is None:
        return _new_state(sim)
    return state


def run_steps(sim, count: int) -> None:
    """Advance device fields by ``count`` steps without copying them to host.

    ``sim.time`` and ``sim.step_count`` intentionally remain untouched; the
    owning solver updates those public counters after this function returns.
    """

    if not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("count must be a positive integer.")

    state = _ensure_state(sim)
    blocks = _blocks_2d(sim)
    characteristic = sim.radial_boundary == "characteristic"
    for _ in range(int(count)):
        _damp_electric_and_apply_boundary[blocks, THREADS_2D](
            state["Er"],
            state["Ephi"],
            state["Hz"],
            state["damp_er_half"],
            state["damp_ephi_half"],
            characteristic,
        )
        _update_magnetic[blocks, THREADS_2D](
            state["Hz"],
            state["Er"],
            state["Ephi"],
            state["rho_faces"],
            state["rho_centers"],
            state["n_hz"],
            state["damp_h_half"],
            sim.drho,
            sim.dphi,
            sim.dt,
        )
        _update_electric[blocks, THREADS_2D](
            state["Er"],
            state["Ephi"],
            state["Hz"],
            state["rho_centers"],
            state["n_er"],
            state["n_ephi"],
            sim.drho,
            sim.dphi,
            sim.dt,
        )
        _damp_electric_and_apply_boundary[blocks, THREADS_2D](
            state["Er"],
            state["Ephi"],
            state["Hz"],
            state["damp_er_half"],
            state["damp_ephi_half"],
            characteristic,
        )

    state["host_dirty"] = True
    sim._gpu_host_dirty = True
    stats = sim._gpu_transfer_stats
    stats["steps"] = int(stats.get("steps", 0)) + int(count)


def sync_to_host(sim) -> bool:
    """Copy dirty device fields into the solver's existing NumPy arrays.

    Returns ``True`` when a transfer occurred and ``False`` when no CUDA state
    exists or the host fields were already current.
    """

    state = _get_state(sim)
    if state is None or not bool(state.get("host_dirty", False)):
        return False

    state["Hz"].copy_to_host(sim.Hz)
    state["Er"].copy_to_host(sim.Er)
    state["Ephi"].copy_to_host(sim.Ephi)
    cuda.synchronize()
    state["host_dirty"] = False
    sim._gpu_host_dirty = False
    stats = sim._gpu_transfer_stats
    stats["device_to_host"] = int(stats.get("device_to_host", 0)) + 3
    stats["device_to_host_sync"] = int(stats.get("device_to_host_sync", 0)) + 3
    stats["host_syncs"] = int(stats.get("host_syncs", 0)) + 1
    return True


def discard_state(sim) -> None:
    """Drop device-resident state without copying it back to the host."""

    state = _get_state(sim)
    setattr(sim, _STATE_ATTRIBUTE, None)
    if getattr(sim, "_gpu_state", None) is state:
        sim._gpu_state = None
    sim._gpu_host_dirty = False


def transfer_stats(sim) -> Dict[str, int]:
    """Return a copy of the CUDA transfer/step counters for ``sim``."""

    return dict(getattr(sim, "_gpu_transfer_stats", {}))


def has_state(sim) -> bool:
    """Return whether ``sim`` currently owns device-resident GR fields."""

    return _get_state(sim) is not None


def host_is_dirty(sim) -> bool:
    """Return whether device fields are newer than the NumPy field arrays."""

    state = _get_state(sim)
    return bool(state is not None and state.get("host_dirty", False))


__all__ = [
    "availability",
    "cuda_available",
    "discard_state",
    "has_state",
    "host_is_dirty",
    "is_available",
    "run_steps",
    "sync_to_host",
    "transfer_stats",
]
