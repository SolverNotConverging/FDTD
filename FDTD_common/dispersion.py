"""Rasterization helpers and an auxiliary-differential-equation E update."""

from dataclasses import dataclass

import numpy as np


@dataclass
class PoleField:
    """One dynamics channel with a spatial forcing-strength array."""

    kind: str
    parameters: tuple
    strength: np.ndarray

    @property
    def key(self):
        return self.kind, self.parameters


def material_pole_channels(material, component):
    """Yield ``(kind, dynamics_parameters, forcing_strength)`` for one axis."""
    if material is None:
        return
    for pole in getattr(material, "debye", ()):
        strength = pole.delta_epsilon[component]
        if strength:
            yield "debye", (pole.tau[component],), strength
    for pole in getattr(material, "drude", ()):
        omega_p = pole.omega_p[component]
        if omega_p:
            yield "drude", (pole.gamma[component],), omega_p * omega_p
    for pole in getattr(material, "lorentz", ()):
        delta_epsilon = pole.delta_epsilon[component]
        if delta_epsilon:
            omega_0 = pole.omega_0[component]
            yield ("lorentz", (omega_0, pole.gamma[component]),
                   delta_epsilon * omega_0 * omega_0)


def blend_pole_fields(pole_fields, cell_shape, index, fraction, material, component):
    """Convexly paint a material's pole strengths into cell-centered channels.

    Existing strengths are attenuated by ``1-fraction``.  New poles are merged
    only when their dynamics match exactly; relaxation and resonance rates are
    therefore never averaged across materials.
    """
    weight = np.asarray(fraction, dtype=float)
    for channel in pole_fields:
        current = channel.strength[index]
        channel.strength[index] = current * (1.0 - weight)

    by_key = {channel.key: channel for channel in pole_fields}
    for kind, parameters, strength in material_pole_channels(material, component):
        key = kind, parameters
        channel = by_key.get(key)
        if channel is None:
            channel = PoleField(kind, parameters, np.zeros(cell_shape, dtype=float))
            pole_fields.append(channel)
            by_key[key] = channel
        current = channel.strength[index]
        channel.strength[index] = current + weight * strength

    pole_fields[:] = [channel for channel in pole_fields
                      if np.any(channel.strength != 0.0)]


def average_pole_fields(pole_fields, mapper):
    """Map cell-centered forcing strengths to one Yee component."""
    return [PoleField(channel.kind, channel.parameters,
                      np.ascontiguousarray(mapper(channel.strength)))
            for channel in pole_fields]


class ADEState:
    """Centered, passive multipole ADE constitutive state for one E component.

    Polarization is stored as ``q=P/epsilon_0``.  Debye is trapezoidal in q;
    Drude and Lorentz use a trapezoidal first-order q/velocity system.  The
    latter is a bilinear (A-stable) discretization and does not impose an
    additional explicit material-pole time-step limit.
    """

    def __init__(self, epsilon_r, sigma_e, eps0, dt, curl_scale,
                 pole_fields=(), mask=None):
        eps0 = float(eps0)
        dt = float(dt)
        curl_scale = float(curl_scale)
        if not np.isfinite(eps0) or eps0 <= 0.0:
            raise ValueError("eps0 must be finite and positive.")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive.")
        if not np.isfinite(curl_scale):
            raise ValueError("curl_scale must be finite.")
        epsilon = np.asarray(epsilon_r, dtype=float)
        if not np.all(np.isfinite(epsilon)) or np.any(epsilon <= 0.0):
            raise ValueError("ADE epsilon_r must be finite and positive.")
        try:
            conductivity = np.broadcast_to(
                np.asarray(sigma_e, dtype=float), epsilon.shape)
        except ValueError as exc:
            raise ValueError("ADE sigma_e must broadcast to the electric field shape.") from exc
        if not np.all(np.isfinite(conductivity)) or np.any(conductivity < 0.0):
            raise ValueError("ADE sigma_e must be finite and non-negative.")

        self.epsilon_r = np.ascontiguousarray(epsilon)
        self.sigma_term = np.ascontiguousarray(
            conductivity * dt / (2.0 * eps0))
        self.dt = dt
        self.curl_scale = curl_scale
        self.mask = (np.zeros_like(self.epsilon_r, dtype=bool) if mask is None
                     else np.asarray(mask, dtype=bool).copy())
        if self.mask.shape != self.epsilon_r.shape:
            raise ValueError("ADE conductor mask must match the electric field shape.")

        self.debye = []
        self.oscillators = []
        implicit = np.zeros_like(self.epsilon_r)
        for channel in pole_fields:
            forcing = np.ascontiguousarray(np.asarray(channel.strength, dtype=float))
            if forcing.shape != self.epsilon_r.shape:
                raise ValueError("ADE pole strength must match the electric field shape.")
            if not np.all(np.isfinite(forcing)) or np.any(forcing < 0.0):
                raise ValueError("ADE pole strength must be finite and non-negative.")
            if channel.kind == "debye":
                if len(channel.parameters) != 1:
                    raise ValueError("A Debye ADE channel requires (tau,).")
                tau = float(channel.parameters[0])
                if not np.isfinite(tau) or tau <= 0.0:
                    raise ValueError("Debye tau must be finite and positive.")
                a = (2.0 * tau - self.dt) / (2.0 * tau + self.dt)
                r = forcing * self.dt / (2.0 * tau + self.dt)
                self.debye.append({
                    "kind": "debye", "parameters": channel.parameters,
                    "a": a, "r": r, "q": np.zeros_like(self.epsilon_r),
                })
            else:
                if channel.kind == "drude":
                    if len(channel.parameters) != 1:
                        raise ValueError("A Drude ADE channel requires (gamma,).")
                    omega_0, gamma = 0.0, float(channel.parameters[0])
                elif channel.kind == "lorentz":
                    if len(channel.parameters) != 2:
                        raise ValueError(
                            "A Lorentz ADE channel requires (omega_0, gamma).")
                    omega_0, gamma = map(float, channel.parameters)
                else:
                    raise ValueError(f"Unknown ADE pole kind {channel.kind!r}.")
                if not np.isfinite(gamma) or gamma < 0.0:
                    raise ValueError("ADE damping must be finite and non-negative.")
                if (channel.kind == "lorentz"
                        and (not np.isfinite(omega_0) or omega_0 <= 0.0)):
                    raise ValueError("Lorentz omega_0 must be finite and positive.")
                half = 0.5 * self.dt
                denominator = 1.0 + gamma * half + (omega_0 * half) ** 2
                a = (1.0 + gamma * half - (omega_0 * half) ** 2) / denominator
                b = self.dt / denominator
                r = forcing * half * half / denominator
                self.oscillators.append({
                    "kind": channel.kind, "parameters": channel.parameters,
                    "a": a, "b": b, "r": r,
                    "q": np.zeros_like(self.epsilon_r),
                    "v": np.zeros_like(self.epsilon_r),
                })
            implicit += r

        self.implicit = np.ascontiguousarray(implicit)
        denominator = self.epsilon_r + self.sigma_term + self.implicit
        self.denominator = np.ascontiguousarray(denominator)
        self.ca = np.ascontiguousarray(
            (self.epsilon_r - self.sigma_term - self.implicit) / denominator)
        self.cb = np.ascontiguousarray(self.curl_scale / denominator)
        self.ca[self.mask] = 0.0
        self.cb[self.mask] = 0.0
        self._dispersive = any(
            np.any((pole["r"] != 0.0) & ~self.mask) for pole in self.poles)

    @property
    def dispersive(self):
        return self._dispersive

    @property
    def poles(self):
        return tuple(self.debye + self.oscillators)

    def displacement(self, field, index=None):
        """Return normalized total displacement ``epsilon_inf*E + sum(q)``."""
        region = Ellipsis if index is None else index
        value = self.epsilon_r[region] * np.asarray(field)[region]
        value = np.array(value, copy=True)
        for pole in self.poles:
            value += pole["q"][region]
        return value

    def solve_displacement(self, field, trial_displacement, index=None):
        """Solve the coupled constitutive equation and advance every ADE pole."""
        region = Ellipsis if index is None else index
        old_e = np.array(np.asarray(field)[region], copy=True)
        rhs = (np.asarray(trial_displacement)
               - self.sigma_term[region] * old_e)

        for pole in self.debye:
            rhs -= (pole["a"] * pole["q"][region]
                    + pole["r"][region] * old_e)
        for pole in self.oscillators:
            rhs -= (pole["a"] * pole["q"][region]
                    + pole["b"] * pole["v"][region]
                    + pole["r"][region] * old_e)

        new_e = rhs / self.denominator[region]
        local_mask = self.mask[region]
        if np.any(local_mask):
            new_e = np.where(local_mask, 0.0, new_e)

        field[region] = new_e
        field_sum = new_e + old_e
        for pole in self.debye:
            q_new = (pole["a"] * pole["q"][region]
                     + pole["r"][region] * field_sum)
            if np.any(local_mask):
                q_new = np.where(local_mask, 0.0, q_new)
            pole["q"][region] = q_new
        for pole in self.oscillators:
            q_old = np.array(pole["q"][region], copy=True)
            v_old = np.array(pole["v"][region], copy=True)
            q_new = (pole["a"] * q_old + pole["b"] * v_old
                     + pole["r"][region] * field_sum)
            v_new = 2.0 * (q_new - q_old) / self.dt - v_old
            if np.any(local_mask):
                q_new = np.where(local_mask, 0.0, q_new)
                v_new = np.where(local_mask, 0.0, v_new)
            pole["q"][region] = q_new
            pole["v"][region] = v_new
        return self.displacement(field, region)

    def advance(self, field, curl, index=None):
        """Advance E and all poles from a curl value at the selected locations."""
        region = Ellipsis if index is None else index
        trial = self.displacement(field, region) + self.curl_scale * np.asarray(curl)
        return self.solve_displacement(field, trial, region)

    def correct_displacement(self, field, delta_trial, index=None):
        """Apply a same-time displacement correction to a completed ADE step.

        This linear correction is useful when a public update has already been
        finalized and a caller then adds a soft source to total displacement.
        It adjusts E, q, and v without advancing the pole dynamics twice.
        """
        region = Ellipsis if index is None else index
        delta_e = np.asarray(delta_trial) / self.denominator[region]
        local_mask = self.mask[region]
        if np.any(local_mask):
            delta_e = np.where(local_mask, 0.0, delta_e)
        field[region] += delta_e
        for pole in self.debye:
            pole["q"][region] += pole["r"][region] * delta_e
        for pole in self.oscillators:
            delta_q = pole["r"][region] * delta_e
            pole["q"][region] += delta_q
            pole["v"][region] += 2.0 * delta_q / self.dt
        return self.displacement(field, region)

    def advance_imposed(self, field, old_field, index=None):
        """Advance pole histories for an externally imposed new E endpoint."""
        region = Ellipsis if index is None else index
        new_e = np.asarray(field)[region]
        old_e = np.asarray(old_field)
        field_sum = new_e + old_e
        local_mask = self.mask[region]
        for pole in self.debye:
            q_new = (pole["a"] * pole["q"][region]
                     + pole["r"][region] * field_sum)
            pole["q"][region] = np.where(local_mask, 0.0, q_new)
        for pole in self.oscillators:
            q_old = np.array(pole["q"][region], copy=True)
            v_old = np.array(pole["v"][region], copy=True)
            q_new = (pole["a"] * q_old + pole["b"] * v_old
                     + pole["r"][region] * field_sum)
            v_new = 2.0 * (q_new - q_old) / self.dt - v_old
            pole["q"][region] = np.where(local_mask, 0.0, q_new)
            pole["v"][region] = np.where(local_mask, 0.0, v_new)
        return self.displacement(field, region)

    def copy_history_from(self, previous):
        """Copy compatible polarization state after a material-grid rebuild.

        Histories with matching dynamics are preserved. If a pole's painted
        strength changed, q and v are scaled by the new/old strength ratio;
        newly added, removed, and conductor-masked locations start at zero.
        """
        if not isinstance(previous, ADEState) or previous.dt != self.dt:
            return self
        previous_by_key = {
            (pole["kind"], pole["parameters"]): pole
            for pole in previous.poles
        }
        for pole in self.poles:
            old = previous_by_key.get((pole["kind"], pole["parameters"]))
            if old is None or old["q"].shape != pole["q"].shape:
                continue
            scale = np.zeros_like(pole["r"])
            np.divide(pole["r"], old["r"], out=scale, where=old["r"] != 0.0)
            active = (pole["r"] != 0.0) & ~self.mask
            pole["q"][active] = old["q"][active] * scale[active]
            if "v" in pole and "v" in old:
                pole["v"][active] = old["v"][active] * scale[active]
        return self

    def reset(self):
        """Zero every polarization and polarization-velocity history."""
        for pole in self.debye:
            pole["q"].fill(0.0)
        for pole in self.oscillators:
            pole["q"].fill(0.0)
            pole["v"].fill(0.0)
