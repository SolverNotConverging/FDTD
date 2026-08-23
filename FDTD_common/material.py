"""Material and dispersive-pole definitions shared by every solver."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


def as_triple(value, name, positive=False, nonnegative=False):
    """Normalize a scalar or Cartesian three-vector to a validated tuple."""
    values = np.asarray(
        value if np.ndim(value) else (value, value, value), dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite scalar or length-three sequence.")
    if positive and np.any(values <= 0):
        raise ValueError(f"{name} must be positive.")
    if nonnegative and np.any(values < 0):
        raise ValueError(f"{name} must be non-negative.")
    return tuple(float(item) for item in values)


@dataclass(frozen=True)
class DebyePole:
    """One Debye relaxation pole (``tau`` is in seconds)."""

    delta_epsilon: tuple
    tau: tuple

    def __post_init__(self):
        object.__setattr__(self, "delta_epsilon", as_triple(
            self.delta_epsilon, "Debye delta_epsilon", nonnegative=True))
        object.__setattr__(self, "tau", as_triple(
            self.tau, "Debye tau", positive=True))


@dataclass(frozen=True)
class DrudePole:
    """One Drude pole; ``omega_p`` and ``gamma`` are in radians/second."""

    omega_p: tuple
    gamma: tuple = (0.0, 0.0, 0.0)

    def __post_init__(self):
        object.__setattr__(self, "omega_p", as_triple(
            self.omega_p, "Drude omega_p", nonnegative=True))
        object.__setattr__(self, "gamma", as_triple(
            self.gamma, "Drude gamma", nonnegative=True))


@dataclass(frozen=True)
class LorentzPole:
    """One Lorentz pole; ``omega_0`` and ``gamma`` are in radians/second."""

    delta_epsilon: tuple
    omega_0: tuple
    gamma: tuple = (0.0, 0.0, 0.0)

    def __post_init__(self):
        object.__setattr__(self, "delta_epsilon", as_triple(
            self.delta_epsilon, "Lorentz delta_epsilon", nonnegative=True))
        object.__setattr__(self, "omega_0", as_triple(
            self.omega_0, "Lorentz omega_0", positive=True))
        object.__setattr__(self, "gamma", as_triple(
            self.gamma, "Lorentz gamma", nonnegative=True))


_POLE_FIELDS = {
    DebyePole: ("delta_epsilon", "tau"),
    DrudePole: ("omega_p", "gamma"),
    LorentzPole: ("delta_epsilon", "omega_0", "gamma"),
}

_POLE_ALIASES = {
    DebyePole: {
        "delta_eps": "delta_epsilon", "strength": "delta_epsilon",
        "relaxation_time": "tau",
    },
    DrudePole: {
        "plasma_frequency": "omega_p", "plasma_angular_frequency": "omega_p",
        "plasma_freq": "omega_p", "omegap": "omega_p", "wp": "omega_p",
        "collision_frequency": "gamma", "collision_freq": "gamma",
        "damping": "gamma",
    },
    LorentzPole: {
        "delta_eps": "delta_epsilon", "strength": "delta_epsilon",
        "resonance_frequency": "omega_0", "resonance_angular_frequency": "omega_0",
        "resonance_freq": "omega_0", "omega0": "omega_0", "w0": "omega_0",
        "damping": "gamma",
    },
}


def _pole_from_mapping(value, pole_type, label):
    aliases = _POLE_ALIASES[pole_type]
    normalized = {}
    for key, item in value.items():
        canonical = aliases.get(str(key), str(key))
        if canonical in normalized:
            raise ValueError(f"{label} pole specifies {canonical!r} more than once.")
        normalized[canonical] = item
    unknown = set(normalized).difference(_POLE_FIELDS[pole_type])
    if unknown:
        raise ValueError(f"Unknown {label} pole parameter(s): {sorted(unknown)}.")
    try:
        return pole_type(**normalized)
    except TypeError as exc:
        raise ValueError(f"Invalid {label} pole parameters: {exc}.") from exc


def normalize_poles(value, pole_type, label):
    """Normalize one pole or a sequence of poles to an immutable tuple."""
    if value is None:
        return ()
    if isinstance(value, pole_type):
        return (value,)
    if isinstance(value, Mapping):
        return (_pole_from_mapping(value, pole_type, label),)
    if (isinstance(value, (str, bytes))
            or not isinstance(value, (Sequence, np.ndarray))):
        raise TypeError(f"{label} must be a pole, mapping, or sequence of poles.")

    items = list(value)
    field_count = len(_POLE_FIELDS[pole_type])
    # A short positional sequence is a convenient single-pole form. A list of
    # pole objects, mappings, or nested tuples remains the multipole form.
    nested_parameters = any(
        isinstance(item, (Sequence, np.ndarray))
        and not isinstance(item, (str, bytes))
        for item in items
    )
    # Only a flat scalar sequence denotes one positional pole. Nested
    # positional entries always denote multiple isotropic poles, avoiding the
    # otherwise ambiguous 3x3 Lorentz case. Use a mapping or Pole object for
    # one anisotropic pole.
    positional_single = not nested_parameters
    if (len(items) == field_count and positional_single
            and not any(isinstance(item, (pole_type, Mapping)) for item in items)):
        try:
            return (pole_type(*items),)
        except (TypeError, ValueError):
            pass

    poles = []
    for item in items:
        if isinstance(item, pole_type):
            poles.append(item)
        elif isinstance(item, Mapping):
            poles.append(_pole_from_mapping(item, pole_type, label))
        elif (isinstance(item, (Sequence, np.ndarray))
              and not isinstance(item, (str, bytes))
              and len(item) == field_count):
            poles.append(pole_type(*item))
        else:
            raise TypeError(f"Every {label} entry must describe one {label} pole.")
    return tuple(poles)


@dataclass(frozen=True)
class Material:
    """Diagonal electromagnetic material with optional electric dispersion.

    ``epsilon_r`` is the instantaneous/high-frequency relative permittivity.
    Debye, Drude, and Lorentz pole collections may be used simultaneously.
    """

    name: str
    epsilon_r: tuple = (1.0, 1.0, 1.0)
    mu_r: tuple = (1.0, 1.0, 1.0)
    sigma_e: tuple = (0.0, 0.0, 0.0)
    sigma_m: tuple = (0.0, 0.0, 0.0)
    kind: str = "ordinary"
    debye: tuple = ()
    drude: tuple = ()
    lorentz: tuple = ()

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Material name must be a non-empty string.")
        kind = str(self.kind).upper()
        if kind not in {"ORDINARY", "PEC", "PMC"}:
            raise ValueError("kind must be ordinary, PEC, or PMC.")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(
            self, "epsilon_r", as_triple(self.epsilon_r, "epsilon_r", positive=True))
        object.__setattr__(
            self, "mu_r", as_triple(self.mu_r, "mu_r", positive=True))
        object.__setattr__(
            self, "sigma_e", as_triple(self.sigma_e, "sigma_e", nonnegative=True))
        object.__setattr__(
            self, "sigma_m", as_triple(self.sigma_m, "sigma_m", nonnegative=True))
        object.__setattr__(self, "debye", normalize_poles(
            self.debye, DebyePole, "Debye"))
        object.__setattr__(self, "drude", normalize_poles(
            self.drude, DrudePole, "Drude"))
        object.__setattr__(self, "lorentz", normalize_poles(
            self.lorentz, LorentzPole, "Lorentz"))
        if kind != "ORDINARY" and (self.debye or self.drude or self.lorentz):
            raise ValueError("PEC and PMC materials cannot contain dispersive poles.")
        object.__setattr__(self, "kind", "ordinary" if kind == "ORDINARY" else kind)

    @property
    def has_dispersion(self):
        return bool(self.debye or self.drude or self.lorentz)

    @property
    def debye_poles(self):
        return self.debye

    @property
    def drude_poles(self):
        return self.drude

    @property
    def lorentz_poles(self):
        return self.lorentz

    def relative_permittivity(self, omega):
        """Evaluate the diagonal complex pole relative permittivity.

        ``omega`` is an angular frequency in radians per second.  The returned
        array has shape ``omega.shape + (3,)`` and uses the ``exp(-i*omega*t)``
        phasor convention. Electric conductivity remains a separate material
        term and is not folded into this value.
        """
        angular_frequency = np.asarray(omega, dtype=float)
        if not np.all(np.isfinite(angular_frequency)):
            raise ValueError("omega must contain only finite angular frequencies.")
        result = np.empty(angular_frequency.shape + (3,), dtype=complex)
        result[...] = np.asarray(self.epsilon_r, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            for pole in self.debye:
                result += np.asarray(pole.delta_epsilon) / (
                    1.0 - 1j * angular_frequency[..., None] * np.asarray(pole.tau))
            for pole in self.drude:
                frequency = angular_frequency[..., None]
                numerator = np.broadcast_to(
                    np.asarray(pole.omega_p) ** 2, result.shape)
                denominator = (frequency ** 2
                               + 1j * frequency * np.asarray(pole.gamma))
                contribution = np.zeros_like(result)
                np.divide(numerator, denominator, out=contribution,
                          where=numerator != 0.0)
                result -= contribution
            for pole in self.lorentz:
                frequency = angular_frequency[..., None]
                omega_0 = np.asarray(pole.omega_0)
                result += (np.asarray(pole.delta_epsilon) * omega_0 ** 2 / (
                    omega_0 ** 2 - frequency ** 2
                    - 1j * frequency * np.asarray(pole.gamma)))
        return result

    epsilon_r_at_omega = relative_permittivity
