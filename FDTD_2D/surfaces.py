"""Passive scalar SIBC, including exact PEC/PMC limits and Foster memory.

Z(s) = R + sum(r*s/(s+p)), r >= 0, p > 0. Positive-real by construction.
The exp(+i omega t) convention is used by ports and Fourier accumulation.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SurfaceImpedance:
    resistance: float = 0.0
    poles: tuple = ()  # (resistance, angular relaxation rate)
    name: str = "surface"

    def __post_init__(self):
        if np.isnan(self.resistance) or self.resistance < 0:
            raise ValueError("Surface resistance must be nonnegative or +inf")
        poles = tuple(tuple(float(v) for v in pair) for pair in self.poles)
        if any(len(v) != 2 or not np.all(np.isfinite(v)) or v[0] < 0 or v[1] <= 0 for v in poles):
            raise ValueError("Foster poles require nonnegative resistance and positive rate")
        if np.isinf(self.resistance) and poles:
            raise ValueError("PMC cannot have dynamic poles")
        object.__setattr__(self, "poles", poles)

    @property
    def is_pec(self):
        return self.resistance == 0 and not self.poles

    @property
    def is_pmc(self):
        return np.isposinf(self.resistance)

    def impedance(self, frequencies, dt=None):
        f = np.asarray(frequencies, dtype=float)
        if np.any(f < 0) or not np.all(np.isfinite(f)):
            raise ValueError("Frequencies must be finite and nonnegative")
        if dt is not None:
            if dt <= 0 or np.any(f * dt >= .5):
                raise ValueError("Frequency outside temporal Nyquist")
            s = 2j / dt * np.tan(np.pi * f * dt)
        else:
            s = 2j * np.pi * f
        z = np.full(f.shape, self.resistance, dtype=complex)
        for r, rate in self.poles:
            z += r * s / (s + rate)
        return z

    def state_space(self, admittance=False):
        """Return a real realization, optionally the exact inverse transfer law."""
        rates = np.array([p for _, p in self.poles])
        r = np.array([r for r, _ in self.poles])
        a, b, c = -np.diag(rates), rates, -r
        d = self.resistance + r.sum()
        if admittance:
            if d <= 0 or not np.isfinite(d):
                raise ValueError("Handle PEC/PMC as exact topology limits")
            a = a - np.outer(b, c) / d
            b, c, d = b / d, -c / d, 1 / d
        return a, b, c, d

    @classmethod
    def good_conductor(cls, conductivity, band, order=16, tolerance=.05, name="metal"):
        """Fit sqrt(s*mu0/sigma) over an explicit band using nonnegative Foster terms."""
        from scipy.optimize import nnls
        if conductivity <= 0 or not np.isfinite(conductivity) or not 0 < band[0] < band[1] or order < 2:
            raise ValueError("Invalid conductivity, band or fit order")
        rates = 2*np.pi*np.geomspace(band[0]/100, band[1]*100, order)
        f = np.geomspace(*band, 256)
        s = 2j*np.pi*f
        target = np.sqrt(s * (4e-7*np.pi) / conductivity)
        basis = np.column_stack([np.ones(len(f)), s[:, None]/(s[:, None]+rates)])
        matrix = basis / abs(target[:, None])
        weights, _ = nnls(np.vstack([matrix.real, matrix.imag]),
                          np.r_[target.real/abs(target), target.imag/abs(target)])
        model = cls(weights[0], tuple(zip(weights[1:], rates)), name)
        error = np.max(abs(model.impedance(f)-target)/abs(target))
        if error > tolerance:
            raise ValueError(f"Foster fit error {error:.3g} exceeds {tolerance}; increase order")
        return model


PEC = SurfaceImpedance(0.0, name="PEC")
PMC = SurfaceImpedance(float("inf"), name="PMC")


@dataclass(frozen=True)
class ThinSheet:
    """Reciprocal symmetric two-sided sheet network in even/odd current channels.

    [E_left,E_right] = Z [K_left,K_right], currents directed into the sheet.
    Z = U diag(even,odd) U.T; U = [[1,1],[1,-1]]/sqrt(2).
    Unlike two independent opaque SIBCs, this network permits transmission.
    """
    even: SurfaceImpedance
    odd: SurfaceImpedance
    name: str = "thin sheet"

    @classmethod
    def resistive(cls, sheet_resistance, name="resistive sheet"):
        if not np.isfinite(sheet_resistance) or sheet_resistance < 0:
            raise ValueError("Sheet resistance must be finite and nonnegative")
        return cls(SurfaceImpedance(2*sheet_resistance), PEC, name)

    @classmethod
    def conductive(cls, conductivity, thickness, band, order=64, tolerance=.05, name="thin metal"):
        """Finite-thickness diffusion SIBC, with explicit transmission.

        Foster expansions of Zc*coth(gamma*t/2) and Zc*tanh(gamma*t/2).
        Displacement current inside the good conductor is neglected.
        """
        if conductivity <= 0 or thickness <= 0 or not 0 < band[0] < band[1] or order < 1:
            raise ValueError("Invalid thin conductor parameters")
        mu = 4e-7*np.pi
        r = 4/(conductivity*thickness)
        rate = np.pi**2/(mu*conductivity*thickness**2)
        even = SurfaceImpedance(2/(conductivity*thickness),
                                tuple((r, rate*(2*n)**2) for n in range(1, order+1)))
        odd = SurfaceImpedance(0, tuple((r, rate*(2*n+1)**2) for n in range(order)))
        f = np.geomspace(*band, 128)
        s = 2j*np.pi*f
        zc = np.sqrt(s*mu/conductivity)
        u = thickness/2*np.sqrt(s*mu*conductivity)
        targets = (zc/np.tanh(u), zc*np.tanh(u))
        error = max(np.max(abs(model.impedance(f)-target)/abs(target))
                    for model, target in zip((even, odd), targets))
        if error > tolerance:
            raise ValueError(f"Thin-sheet fit error {error:.3g}; increase order")
        return cls(even, odd, name)
