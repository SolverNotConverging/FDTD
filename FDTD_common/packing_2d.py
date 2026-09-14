"""CUDA-independent source packing for the compiled Cartesian 2D runtimes."""

import math
import numpy as np

class _Events:
    def __init__(self):
        self.target = []
        self.i = []
        self.j = []
        self.source = []
        self.factor = []
        self.shift = []

    def add(self, target, i, j, source, factor, shift, shapes):
        shape = shapes[int(target)]
        i, j = int(i), int(j)
        if not (0 <= i < shape[0] and 0 <= j < shape[1]):
            return
        self.target.append(int(target))
        self.i.append(i)
        self.j.append(j)
        self.source.append(int(source))
        self.factor.append(float(factor))
        self.shift.append(float(shift))

    @property
    def count(self):
        return len(self.target)


class _TableEvents:
    def __init__(self, steps):
        self.steps = int(steps)
        self.target = []
        self.i = []
        self.j = []
        self.values = []

    def add(self, target, i, j, values, shape):
        i, j = int(i), int(j)
        if not (0 <= i < shape[int(target)][0] and 0 <= j < shape[int(target)][1]):
            return
        values = np.asarray(values, dtype=float)
        if values.shape != (self.steps,):
            raise ValueError(
                f"Broadband source event must contain {self.steps} time samples.")
        self.target.append(int(target))
        self.i.append(i)
        self.j.append(j)
        self.values.append(values.copy())

    @property
    def count(self):
        return len(self.target)


def _source_parameters(sources):
    count = max(1, len(sources))
    modes = np.zeros(count, np.int8)
    amplitudes = np.zeros(count, np.float64)
    t0 = np.zeros(count, np.float64)
    tw = np.ones(count, np.float64)
    frequencies = np.zeros(count, np.float64)
    for index, source in enumerate(sources):
        amplitudes[index] = source["amplitude"]
        t0[index] = source["t0"]
        tw[index] = source["tw"]
        fmin, fmax = source["f_min"], source["f_max"]
        if fmin is None:
            modes[index] = 0
        elif np.isclose(fmin, fmax):
            modes[index] = 1
            frequencies[index] = fmax
        else:
            modes[index] = 2
            frequencies[index] = 0.5 * (fmin + fmax)
    return modes, amplitudes, t0, tw, frequencies


def _delay(source, name, offset):
    values = np.asarray(source[name], dtype=float).ravel()
    return float(values[offset])


def _compile_tm_events(sim):
    electric = _Events()
    magnetic = _Events()
    broadband_electric = _TableEvents(sim.Nt)
    broadband_magnetic = _TableEvents(sim.Nt)
    soft = _Events()
    e_shapes = (sim.d_Ez_y.shape, sim.d_Ez_x.shape)
    h_shapes = (sim.d_Hx_y.shape, sim.d_Hy_x.shape)
    soft_shapes = (sim.Dz.shape, sim.Dz.shape)
    for sid, source in enumerate(sim.sources):
        kind = source["kind"]
        ix0, ix1 = source["ix0"], source["ix1"]
        iy0, iy1 = source["iy0"], source["iy1"]
        if kind == "sftf":
            for offset, j in enumerate(range(iy0, iy1 + 1)):
                electric.add(1, ix0 - 1, j, sid, -1.0 / sim.dx,
                             -_delay(source, "Ez_delay_xlo", offset), e_shapes)
                electric.add(1, ix1, j, sid, 1.0 / sim.dx,
                             -_delay(source, "Ez_delay_xhi", offset), e_shapes)
                magnetic.add(1, ix0, j, sid, math.cos(source["angle"]) / sim.dx,
                             sim.dt / 2 - _delay(source, "Hy_delay_xlo", offset), h_shapes)
                magnetic.add(1, ix1 + 1, j, sid, -math.cos(source["angle"]) / sim.dx,
                             sim.dt / 2 - _delay(source, "Hy_delay_xhi", offset), h_shapes)
            for offset, i in enumerate(range(ix0, ix1 + 1)):
                electric.add(0, i, iy0 - 1, sid, -1.0 / sim.dy,
                             -_delay(source, "Ez_delay_ylo", offset), e_shapes)
                electric.add(0, i, iy1, sid, 1.0 / sim.dy,
                             -_delay(source, "Ez_delay_yhi", offset), e_shapes)
                magnetic.add(0, i, iy0, sid, -math.sin(source["angle"]) / sim.dy,
                             sim.dt / 2 - _delay(source, "Hx_delay_ylo", offset), h_shapes)
                magnetic.add(0, i, iy1 + 1, sid, math.sin(source["angle"]) / sim.dy,
                             sim.dt / 2 - _delay(source, "Hx_delay_yhi", offset), h_shapes)
        elif kind == "waveguide-y":
            lo, hi = sorted((ix0, ix1))
            if source.get("broadband", False):
                for offset, i in enumerate(range(lo, hi + 1)):
                    broadband_electric.add(
                        0, i, iy0 - 1,
                        -source["broadband_electric_drive"][:, offset] / sim.dy,
                        e_shapes)
                    broadband_magnetic.add(
                        0, i, iy0,
                        source["broadband_magnetic_drive"][:, offset] / sim.dy,
                        h_shapes)
            else:
                for i in range(lo, hi + 1):
                    electric.add(0, i, iy0 - 1, sid,
                                 -source["Ez_src"][i - lo] / sim.dy, 0.0, e_shapes)
                    magnetic.add(0, i, iy0, sid,
                                 source["Hx_src"][i - lo] / sim.dy,
                                 sim.dt / 2 + sim.dy * source["n_eff"] / (2 * sim.c0), h_shapes)
        elif kind == "waveguide-x":
            lo, hi = sorted((iy0, iy1))
            if source.get("broadband", False):
                for offset, j in enumerate(range(lo, hi + 1)):
                    broadband_electric.add(
                        1, ix0 - 1, j,
                        -source["broadband_electric_drive"][:, offset] / sim.dx,
                        e_shapes)
                    broadband_magnetic.add(
                        1, ix0, j,
                        -source["broadband_magnetic_drive"][:, offset] / sim.dx,
                        h_shapes)
            else:
                for j in range(lo, hi + 1):
                    electric.add(1, ix0 - 1, j, sid,
                                 -source["Ez_src"][j - lo] / sim.dx, 0.0, e_shapes)
                    magnetic.add(1, ix0, j, sid,
                                 -source["Hy_src"][j - lo] / sim.dx,
                                 sim.dt / 2 + sim.dx * source["n_eff"] / (2 * sim.c0), h_shapes)
        elif kind == "point":
            soft.add(0, ix0, iy0, sid, 1.0, 0.0, soft_shapes)
        elif kind == "line-soft":
            if ix0 != ix1:
                for i in range(min(ix0, ix1), max(ix0, ix1)):
                    soft.add(0, i, iy0, sid, 1.0, 0.0, soft_shapes)
            else:
                for j in range(min(iy0, iy1), max(iy0, iy1)):
                    soft.add(0, ix0, j, sid, 1.0, 0.0, soft_shapes)
    return electric, magnetic, broadband_electric, broadband_magnetic, soft


def _compile_te_events(sim):
    electric = _Events()
    magnetic = _Events()
    broadband_electric = _TableEvents(sim.Nt)
    broadband_magnetic = _TableEvents(sim.Nt)
    soft = _Events()
    e_shapes = (sim.d_Ex_y.shape, sim.d_Ey_x.shape)
    h_shapes = (sim.d_Hz_y.shape, sim.d_Hz_x.shape)
    soft_shapes = (sim.Bz.shape, sim.Bz.shape)
    for sid, source in enumerate(sim.sources):
        kind = source["kind"]
        ix0, ix1 = source["ix0"], source["ix1"]
        iy0, iy1 = source["iy0"], source["iy1"]
        if kind == "sftf":
            kx, ky = math.cos(source["angle"]), math.sin(source["angle"])
            for offset, j in enumerate(range(iy0, iy1 + 1)):
                electric.add(1, ix0 - 1, j, sid, -kx / sim.dx,
                             -_delay(source, "Ey_delay_xlo", offset), e_shapes)
                electric.add(1, ix1, j, sid, kx / sim.dx,
                             -_delay(source, "Ey_delay_xhi", offset), e_shapes)
                magnetic.add(1, ix0, j, sid, -1.0 / sim.dx,
                             sim.dt / 2 - _delay(source, "Hz_delay_xlo", offset), h_shapes)
                magnetic.add(1, ix1 + 1, j, sid, 1.0 / sim.dx,
                             sim.dt / 2 - _delay(source, "Hz_delay_xhi", offset), h_shapes)
            for offset, i in enumerate(range(ix0, ix1 + 1)):
                electric.add(0, i, iy0 - 1, sid, ky / sim.dy,
                             -_delay(source, "Ex_delay_ylo", offset), e_shapes)
                electric.add(0, i, iy1, sid, -ky / sim.dy,
                             -_delay(source, "Ex_delay_yhi", offset), e_shapes)
                magnetic.add(0, i, iy0, sid, -1.0 / sim.dy,
                             sim.dt / 2 - _delay(source, "Hz_delay_ylo", offset), h_shapes)
                magnetic.add(0, i, iy1 + 1, sid, 1.0 / sim.dy,
                             sim.dt / 2 - _delay(source, "Hz_delay_yhi", offset), h_shapes)
        elif kind == "waveguide-y":
            lo, hi = sorted((ix0, ix1))
            if source.get("broadband", False):
                for offset, i in enumerate(range(lo, hi)):
                    broadband_electric.add(
                        0, i, iy0 - 1,
                        -source["broadband_electric_drive"][:, offset] / sim.dy,
                        e_shapes)
                    broadband_magnetic.add(
                        0, i, iy0,
                        -source["broadband_magnetic_drive"][:, offset] / sim.dy,
                        h_shapes)
            else:
                for i in range(lo, hi):
                    electric.add(0, i, iy0 - 1, sid,
                                 -source["Ex_src"][i - lo] / sim.dy, 0.0, e_shapes)
                    magnetic.add(0, i, iy0, sid,
                                 -source["Hz_src"][i - lo] / sim.dy,
                                 sim.dt / 2 + sim.dy * source["n_eff"] / (2 * sim.c0), h_shapes)
        elif kind == "waveguide-x":
            lo, hi = sorted((iy0, iy1))
            if source.get("broadband", False):
                for offset, j in enumerate(range(lo, hi)):
                    broadband_electric.add(
                        1, ix0 - 1, j,
                        -source["broadband_electric_drive"][:, offset] / sim.dx,
                        e_shapes)
                    broadband_magnetic.add(
                        1, ix0, j,
                        source["broadband_magnetic_drive"][:, offset] / sim.dx,
                        h_shapes)
            else:
                for j in range(lo, hi):
                    electric.add(1, ix0 - 1, j, sid,
                                 -source["Ey_src"][j - lo] / sim.dx, 0.0, e_shapes)
                    magnetic.add(1, ix0, j, sid,
                                 source["Hz_src"][j - lo] / sim.dx,
                                 sim.dt / 2 + sim.dx * source["n_eff"] / (2 * sim.c0), h_shapes)
        elif kind == "point":
            soft.add(0, ix0, iy0, sid, 1.0, 0.0, soft_shapes)
        elif kind == "line-soft":
            if ix0 != ix1:
                for i in range(min(ix0, ix1), max(ix0, ix1)):
                    soft.add(0, i, iy0, sid, 1.0, 0.0, soft_shapes)
            else:
                for j in range(min(iy0, iy1), max(iy0, iy1)):
                    soft.add(0, ix0, j, sid, 1.0, 0.0, soft_shapes)
    return electric, magnetic, broadband_electric, broadband_magnetic, soft
