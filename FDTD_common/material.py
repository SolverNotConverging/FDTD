"""Material definitions shared by the 1D, 2D, and 3D solvers."""

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
class Material:
    """Diagonal electromagnetic material, including optional electric/magnetic loss."""

    name: str
    epsilon_r: tuple = (1.0, 1.0, 1.0)
    mu_r: tuple = (1.0, 1.0, 1.0)
    sigma_e: tuple = (0.0, 0.0, 0.0)
    sigma_m: tuple = (0.0, 0.0, 0.0)
    kind: str = "ordinary"

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Material name must be a non-empty string.")
        kind = str(self.kind).upper()
        if kind not in {"ORDINARY","PEC","PMC"}:
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
        object.__setattr__(self, "kind", "ordinary" if kind == "ORDINARY" else kind)
