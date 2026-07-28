"""Persistent Numba-CUDA runtime shared by the active 2D FDTD solvers.

The CPU-facing solver objects remain NumPy based.  During a GPU run this
module copies immutable coefficients and initial fields to the device once,
keeps every mutable update array resident for the complete time loop, records
requested output on the device, and synchronizes back only after the run.
"""

from __future__ import annotations

import math

import numpy as np
from numba import cuda
from tqdm import tqdm


THREADS_1D = 128
THREADS_2D = (16, 16)


@cuda.jit(device=True, inline=True)
def _waveform(source_id, time, dt, modes, amplitudes, t0, tw, frequencies):
    mode = modes[source_id]
    amplitude = amplitudes[source_id]
    center = t0[source_id]
    width = tw[source_id]
    frequency = frequencies[source_id]
    relative_time = time - center
    if mode == 0:
        ratio = relative_time / width
        return amplitude * math.exp(-(ratio * ratio))
    if mode == 1:
        period = 1.0 / max(frequency, 1e-30)
        ramp_time = max(period, dt)
        elapsed = max(relative_time, 0.0)
        ramp = 1.0 - math.exp(-((elapsed / ramp_time) ** 3))
        return amplitude * ramp * math.sin(2.0 * math.pi * frequency * relative_time)
    ratio = relative_time / width
    return (amplitude * math.sin(2.0 * math.pi * frequency * relative_time)
            * math.exp(-(ratio * ratio)))


@cuda.jit
def _inject_events(first, second, targets, ii, jj, source_ids, factors,
                   shifts, count, step, dt, modes, amplitudes, t0, tw,
                   frequencies):
    event = cuda.grid(1)
    if event >= count:
        return
    value = factors[event] * _waveform(
        source_ids[event], step * dt + shifts[event], dt,
        modes, amplitudes, t0, tw, frequencies)
    index = (ii[event], jj[event])
    if targets[event] == 0:
        cuda.atomic.add(first, index, value)
    else:
        cuda.atomic.add(second, index, value)


@cuda.jit
def _inject_table_events(first, second, targets, ii, jj, values, count, step):
    event = cuda.grid(1)
    if event >= count:
        return
    index = (ii[event], jj[event])
    value = values[event, step]
    if targets[event] == 0:
        cuda.atomic.add(first, index, value)
    else:
        cuda.atomic.add(second, index, value)


@cuda.jit
def _tm_curl_e(ez, d_ez_y, d_ez_x, dx, dy):
    i, j = cuda.grid(2)
    if i < d_ez_y.shape[0] and j < d_ez_y.shape[1]:
        d_ez_y[i, j] = (ez[i, j + 1] - ez[i, j]) / dy
    if i < d_ez_x.shape[0] and j < d_ez_x.shape[1]:
        d_ez_x[i, j] = (ez[i + 1, j] - ez[i, j]) / dx


@cuda.jit
def _tm_update_h(hx, hy, bx, by, hx_previous, hy_previous,
                 d_ez_y, d_ez_x, psi_bx_y, psi_by_x,
                 b_bx_y, c_bx_y, b_by_x, c_by_x,
                 kappa_y_hx, kappa_x_hy, ca_hx, cb_hx, ca_hy, cb_hy,
                 mr_hx, mr_hy, pmc_hx, pmc_hy):
    i, j = cuda.grid(2)
    if i < hx.shape[0] and j < hx.shape[1]:
        old = hx[i, j]
        hx_previous[i, j] = old
        psi = b_bx_y[i, j] * psi_bx_y[i, j] + c_bx_y[i, j] * d_ez_y[i, j]
        psi_bx_y[i, j] = psi
        value = ca_hx[i, j] * old - cb_hx[i, j] * (
            d_ez_y[i, j] / kappa_y_hx[i, j] + psi)
        if pmc_hx[i, j]:
            value = 0.0
        hx[i, j] = value
        bx[i, j] = mr_hx[i, j] * value
    if i < hy.shape[0] and j < hy.shape[1]:
        old = hy[i, j]
        hy_previous[i, j] = old
        psi = b_by_x[i, j] * psi_by_x[i, j] + c_by_x[i, j] * d_ez_x[i, j]
        psi_by_x[i, j] = psi
        value = ca_hy[i, j] * old + cb_hy[i, j] * (
            d_ez_x[i, j] / kappa_x_hy[i, j] + psi)
        if pmc_hy[i, j]:
            value = 0.0
        hy[i, j] = value
        by[i, j] = mr_hy[i, j] * value


@cuda.jit
def _tm_curl_h(hx, hy, d_hx_y, d_hy_x, dx, dy, periodic_x, periodic_y):
    i, j = cuda.grid(2)
    nx = d_hx_y.shape[0] - 1
    ny = d_hx_y.shape[1] - 1
    if i <= nx and j <= ny:
        if j == 0:
            d_hx_y[i, j] = ((hx[i, 0] - hx[i, ny - 1])
                            if periodic_y else hx[i, 0]) / dy
        elif j == ny:
            d_hx_y[i, j] = ((hx[i, 0] - hx[i, ny - 1])
                            if periodic_y else -hx[i, ny - 1]) / dy
        else:
            d_hx_y[i, j] = (hx[i, j] - hx[i, j - 1]) / dy
        if i == 0:
            d_hy_x[i, j] = ((hy[0, j] - hy[nx - 1, j])
                            if periodic_x else hy[0, j]) / dx
        elif i == nx:
            d_hy_x[i, j] = ((hy[0, j] - hy[nx - 1, j])
                            if periodic_x else -hy[nx - 1, j]) / dx
        else:
            d_hy_x[i, j] = (hy[i, j] - hy[i - 1, j]) / dx


@cuda.jit
def _tm_update_e(ez, dz, d_hx_y, d_hy_x, psi_dz_x, psi_dz_y,
                 b_dz_x, c_dz_x, b_dz_y, c_dz_y,
                 kappa_x_ez, kappa_y_ez, ca_ez, cb_ez, er_ez):
    i, j = cuda.grid(2)
    if i < ez.shape[0] and j < ez.shape[1]:
        psi_x = b_dz_x[i, j] * psi_dz_x[i, j] + c_dz_x[i, j] * d_hy_x[i, j]
        psi_y = b_dz_y[i, j] * psi_dz_y[i, j] + c_dz_y[i, j] * d_hx_y[i, j]
        psi_dz_x[i, j] = psi_x
        psi_dz_y[i, j] = psi_y
        value = ca_ez[i, j] * ez[i, j] + cb_ez[i, j] * (
            d_hy_x[i, j] / kappa_x_ez[i, j]
            - d_hx_y[i, j] / kappa_y_ez[i, j] + psi_x - psi_y)
        ez[i, j] = value
        dz[i, j] = er_ez[i, j] * value


@cuda.jit
def _tm_finalize_e(ez, dz, er_ez, pec_ez):
    i, j = cuda.grid(2)
    if i < ez.shape[0] and j < ez.shape[1]:
        if pec_ez[i, j]:
            dz[i, j] = 0.0
            ez[i, j] = 0.0
        else:
            ez[i, j] = dz[i, j] / er_ez[i, j]


@cuda.jit
def _tm_sample_monitors(ez, hx, hy, hx_previous, hy_previous,
                        point_x, point_y, point_offset, point_stride,
                        point_it0, point_it1,
                        monitor_values, point_count, step):
    point = cuda.grid(1)
    if point >= point_count:
        return
    it0 = point_it0[point]
    if step < it0 or step >= point_it1[point]:
        return
    i = point_x[point]
    j = point_y[point]
    output = point_offset[point] + (step - it0) * point_stride[point]
    monitor_values[output, 0] = 0.25 * (
        ez[i, j] + ez[i + 1, j] + ez[i, j + 1] + ez[i + 1, j + 1])
    monitor_values[output, 1] = 0.25 * (
        hx[i, j] + hx_previous[i, j]
        + hx[i + 1, j] + hx_previous[i + 1, j])
    monitor_values[output, 2] = 0.25 * (
        hy[i, j] + hy_previous[i, j]
        + hy[i, j + 1] + hy_previous[i, j + 1])


@cuda.jit
def _tm_record_history(hx, hy, ez, dz, hx_history, hy_history,
                       ez_history, dz_history, record_index):
    i, j = cuda.grid(2)
    if i < hx.shape[0] and j < hx.shape[1]:
        hx_history[record_index, i, j] = hx[i, j]
    if i < hy.shape[0] and j < hy.shape[1]:
        hy_history[record_index, i, j] = hy[i, j]
    if i < ez.shape[0] and j < ez.shape[1]:
        ez_history[record_index, i, j] = ez[i, j]
        dz_history[record_index, i, j] = dz[i, j]


@cuda.jit
def _te_curl_e(ex, ey, d_ex_y, d_ey_x, dx, dy):
    i, j = cuda.grid(2)
    if i < d_ex_y.shape[0] and j < d_ex_y.shape[1]:
        d_ex_y[i, j] = (ex[i, j + 1] - ex[i, j]) / dy
        d_ey_x[i, j] = (ey[i + 1, j] - ey[i, j]) / dx


@cuda.jit
def _te_update_h(hz, bz, hz_previous, d_ex_y, d_ey_x,
                 psi_bz_x, psi_bz_y, b_bz_x, c_bz_x, b_bz_y, c_bz_y,
                 kappa_x, kappa_y, ca_hz, cb_hz, mr_hz):
    i, j = cuda.grid(2)
    if i < hz.shape[0] and j < hz.shape[1]:
        old = hz[i, j]
        hz_previous[i, j] = old
        psi_x = b_bz_x[i, j] * psi_bz_x[i, j] + c_bz_x[i, j] * d_ey_x[i, j]
        psi_y = b_bz_y[i, j] * psi_bz_y[i, j] + c_bz_y[i, j] * d_ex_y[i, j]
        psi_bz_x[i, j] = psi_x
        psi_bz_y[i, j] = psi_y
        value = ca_hz[i, j] * old - cb_hz[i, j] * (
            d_ey_x[i, j] / kappa_x[i, j]
            - d_ex_y[i, j] / kappa_y[i, j] + psi_x - psi_y)
        hz[i, j] = value
        bz[i, j] = mr_hz[i, j] * value


@cuda.jit
def _te_finalize_h(hz, bz, mr_hz, pmc_hz):
    i, j = cuda.grid(2)
    if i < hz.shape[0] and j < hz.shape[1]:
        if pmc_hz[i, j]:
            bz[i, j] = 0.0
            hz[i, j] = 0.0
        else:
            hz[i, j] = bz[i, j] / mr_hz[i, j]


@cuda.jit
def _te_curl_h(hz, d_hz_y, d_hz_x, dx, dy, periodic_x, periodic_y):
    i, j = cuda.grid(2)
    nx, ny = hz.shape
    if i < nx and j <= ny:
        if j == 0:
            d_hz_y[i, j] = ((hz[i, 0] - hz[i, ny - 1])
                            if periodic_y else hz[i, 0]) / dy
        elif j == ny:
            d_hz_y[i, j] = ((hz[i, 0] - hz[i, ny - 1])
                            if periodic_y else -hz[i, ny - 1]) / dy
        else:
            d_hz_y[i, j] = (hz[i, j] - hz[i, j - 1]) / dy
    if i <= nx and j < ny:
        if i == 0:
            d_hz_x[i, j] = ((hz[0, j] - hz[nx - 1, j])
                            if periodic_x else hz[0, j]) / dx
        elif i == nx:
            d_hz_x[i, j] = ((hz[0, j] - hz[nx - 1, j])
                            if periodic_x else -hz[nx - 1, j]) / dx
        else:
            d_hz_x[i, j] = (hz[i, j] - hz[i - 1, j]) / dx


@cuda.jit
def _te_update_e(ex, ey, dx_field, dy_field, d_hz_y, d_hz_x,
                 psi_dx_y, psi_dy_x, b_dx_y, c_dx_y, b_dy_x, c_dy_x,
                 kappa_y_ex, kappa_x_ey, ca_ex, cb_ex, ca_ey, cb_ey,
                 er_ex, er_ey, pec_ex, pec_ey):
    i, j = cuda.grid(2)
    if i < ex.shape[0] and j < ex.shape[1]:
        psi = b_dx_y[i, j] * psi_dx_y[i, j] + c_dx_y[i, j] * d_hz_y[i, j]
        psi_dx_y[i, j] = psi
        value = ca_ex[i, j] * ex[i, j] + cb_ex[i, j] * (
            d_hz_y[i, j] / kappa_y_ex[i, j] + psi)
        if pec_ex[i, j]:
            value = 0.0
        ex[i, j] = value
        dx_field[i, j] = er_ex[i, j] * value
    if i < ey.shape[0] and j < ey.shape[1]:
        psi = b_dy_x[i, j] * psi_dy_x[i, j] + c_dy_x[i, j] * d_hz_x[i, j]
        psi_dy_x[i, j] = psi
        value = ca_ey[i, j] * ey[i, j] - cb_ey[i, j] * (
            d_hz_x[i, j] / kappa_x_ey[i, j] + psi)
        if pec_ey[i, j]:
            value = 0.0
        ey[i, j] = value
        dy_field[i, j] = er_ey[i, j] * value


@cuda.jit
def _te_sample_monitors(ex, ey, hz, hz_previous,
                        point_x, point_y, point_offset, point_stride,
                        point_it0, point_it1,
                        monitor_values, point_count, step):
    point = cuda.grid(1)
    if point >= point_count:
        return
    it0 = point_it0[point]
    if step < it0 or step >= point_it1[point]:
        return
    i = point_x[point]
    j = point_y[point]
    output = point_offset[point] + (step - it0) * point_stride[point]
    monitor_values[output, 0] = 0.5 * (hz[i, j] + hz_previous[i, j])
    monitor_values[output, 1] = 0.5 * (ex[i, j] + ex[i, j + 1])
    monitor_values[output, 2] = 0.5 * (ey[i, j] + ey[i + 1, j])


@cuda.jit
def _te_record_history(ex, ey, hz, ex_history, ey_history, hz_history,
                       record_index):
    i, j = cuda.grid(2)
    if i < ex.shape[0] and j < ex.shape[1]:
        ex_history[record_index, i, j] = ex[i, j]
    if i < ey.shape[0] and j < ey.shape[1]:
        ey_history[record_index, i, j] = ey[i, j]
    if i < hz.shape[0] and j < hz.shape[1]:
        hz_history[record_index, i, j] = hz[i, j]


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


def _to_device(array):
    return cuda.to_device(np.ascontiguousarray(array))


def _device_events(events):
    size = max(1, events.count)
    def padded(values, dtype):
        result = np.zeros(size, dtype=dtype)
        if values:
            result[:len(values)] = values
        return _to_device(result)
    return {
        "target": padded(events.target, np.int8),
        "i": padded(events.i, np.int32),
        "j": padded(events.j, np.int32),
        "source": padded(events.source, np.int32),
        "factor": padded(events.factor, np.float64),
        "shift": padded(events.shift, np.float64),
        "count": events.count,
    }


def _device_table_events(events):
    size = max(1, events.count)

    def padded(values, dtype):
        result = np.zeros(size, dtype=dtype)
        if values:
            result[:len(values)] = values
        return _to_device(result)

    table = np.zeros((size, events.steps), dtype=np.float64)
    if events.values:
        table[:events.count, :] = np.asarray(events.values, dtype=np.float64)
    return {
        "target": padded(events.target, np.int8),
        "i": padded(events.i, np.int32),
        "j": padded(events.j, np.int32),
        "values": _to_device(table),
        "count": events.count,
    }


def _compile_monitors(sim):
    point_x, point_y, point_offset = [], [], []
    point_stride, point_it0, point_it1 = [], [], []
    descriptions = []
    sample_offset = 0
    for monitor in sim.monitors:
        orientation = monitor.get("orientation", "").lower()
        if orientation not in {"horizontal", "vertical"}:
            continue
        ix0, ix1 = int(monitor["ix0"]), int(monitor["ix1"])
        iy0, iy1 = int(monitor["iy0"]), int(monitor["iy1"])
        it0, it1 = int(monitor["it0"]), int(monitor["it1"])
        length = ix1 - ix0 if orientation == "horizontal" else iy1 - iy0
        duration = max(0, it1 - it0)
        if length <= 0 or duration <= 0:
            continue
        for point in range(length):
            point_x.append(ix0 + point if orientation == "horizontal" else ix0)
            point_y.append(iy0 if orientation == "horizontal" else iy0 + point)
            point_offset.append(sample_offset + point)
            point_stride.append(length)
            point_it0.append(it0)
            point_it1.append(it1)
        descriptions.append((dict(monitor), sample_offset, duration, length))
        sample_offset += duration * length
    count = max(1, len(point_x))
    def values(items, dtype):
        result = np.zeros(count, dtype=dtype)
        if items:
            result[:len(items)] = items
        return _to_device(result)
    return {
        "x": values(point_x, np.int32),
        "y": values(point_y, np.int32),
        "offset": values(point_offset, np.int64),
        "stride": values(point_stride, np.int32),
        "it0": values(point_it0, np.int32),
        "it1": values(point_it1, np.int32),
        "values": cuda.device_array((max(1, sample_offset), 3), dtype=np.float64),
        "point_count": len(point_x),
        "sample_count": sample_offset,
        "descriptions": descriptions,
    }


def _event_launch(events, first, second, step, dt, source_device):
    if events["count"] == 0:
        return
    blocks = (events["count"] + THREADS_1D - 1) // THREADS_1D
    _inject_events[blocks, THREADS_1D](
        first, second, events["target"], events["i"], events["j"],
        events["source"], events["factor"], events["shift"], events["count"],
        step, dt, *source_device)


def _table_event_launch(events, first, second, step):
    if events["count"] == 0:
        return
    blocks = (events["count"] + THREADS_1D - 1) // THREADS_1D
    _inject_table_events[blocks, THREADS_1D](
        first, second, events["target"], events["i"], events["j"],
        events["values"], events["count"], step)


def _source_device_arrays(sim):
    return tuple(_to_device(item) for item in _source_parameters(sim.sources))


def _blocks_2d(sim):
    return ((sim.Nx + 1 + THREADS_2D[0] - 1) // THREADS_2D[0],
            (sim.Ny + 1 + THREADS_2D[1] - 1) // THREADS_2D[1])


def _finish_monitors(sim, monitors, names):
    sim.monitor_results = []
    if monitors["sample_count"] == 0:
        return
    values = monitors["values"].copy_to_host()[:monitors["sample_count"]]
    for description, start, duration, length in monitors["descriptions"]:
        sample = values[start:start + duration * length].reshape(duration, length, 3)
        result = dict(description)
        for component, name in enumerate(names):
            result[name] = sample[:, :, component].copy()
        sim.monitor_results.append(result)


def _copy_back(sim, state, names):
    for name in names:
        state[name].copy_to_host(getattr(sim, name))


def run_tm(sim, record_stride=1, is_include_history=True):
    """Run a TMz solver with all per-step state resident on the GPU."""
    sim._init_Coeff()
    stride = int(record_stride)
    if stride < 1:
        raise ValueError("record_stride must be positive.")
    sim.record_stride = stride
    sim.is_include_history = bool(is_include_history)
    state_names = (
        "Ez", "Dz", "Hx", "Hy", "Bx", "By", "d_Ez_y", "d_Ez_x",
        "d_Hx_y", "d_Hy_x", "Psi_Bx_y", "Psi_By_x", "Psi_Dz_x", "Psi_Dz_y",
        "b_Bx_y", "c_Bx_y", "b_By_x", "c_By_x", "b_Dz_x", "c_Dz_x",
        "b_Dz_y", "c_Dz_y", "kappa_y_Hx", "kappa_x_Hy", "kappa_x_Ez",
        "kappa_y_Ez", "CaHx", "CbHx", "CaHy", "CbHy", "CaEz", "CbEz",
        "MRxx_Hx", "MRyy_Hy", "ERzz_Ez", "PMC_Hx", "PMC_Hy", "PEC_Ez",
    )
    state = {name: _to_device(getattr(sim, name)) for name in state_names}
    state["Hx_previous"] = _to_device(sim.Hx)
    state["Hy_previous"] = _to_device(sim.Hy)
    source_device = _source_device_arrays(sim)
    e_host, h_host, be_host, bh_host, soft_host = _compile_tm_events(sim)
    e_events, h_events, soft_events = map(_device_events, (e_host, h_host, soft_host))
    be_events, bh_events = map(_device_table_events, (be_host, bh_host))
    monitors = _compile_monitors(sim)
    record_count = (sim.Nt + stride - 1) // stride if is_include_history else 0
    if is_include_history:
        histories = {
            "Hx": cuda.device_array((record_count,) + sim.Hx.shape, dtype=sim.Hx.dtype),
            "Hy": cuda.device_array((record_count,) + sim.Hy.shape, dtype=sim.Hy.dtype),
            "Ez": cuda.device_array((record_count,) + sim.Ez.shape, dtype=sim.Ez.dtype),
            "Dz": cuda.device_array((record_count,) + sim.Dz.shape, dtype=sim.Dz.dtype),
        }
    else:
        histories = None
    sim._gpu_state = state
    sim._gpu_transfer_stats = {
        "host_to_device_during_steps": 0,
        "device_to_host_during_steps": 0,
        "source_events": (
            e_host.count + h_host.count + be_host.count + bh_host.count + soft_host.count),
        "monitor_points": monitors["point_count"],
    }
    blocks = _blocks_2d(sim)
    periodic_x = "x" in getattr(sim, "periodic", "")
    periodic_y = "y" in getattr(sim, "periodic", "")
    monitor_blocks = ((monitors["point_count"] + THREADS_1D - 1) // THREADS_1D
                      if monitors["point_count"] else 0)
    for step in tqdm(range(sim.Nt), desc="FDTD simulation (GPU resident)", unit="step"):
        _tm_curl_e[blocks, THREADS_2D](
            state["Ez"], state["d_Ez_y"], state["d_Ez_x"], sim.dx, sim.dy)
        _event_launch(e_events, state["d_Ez_y"], state["d_Ez_x"], step, sim.dt, source_device)
        _table_event_launch(be_events, state["d_Ez_y"], state["d_Ez_x"], step)
        _tm_update_h[blocks, THREADS_2D](
            state["Hx"], state["Hy"], state["Bx"], state["By"],
            state["Hx_previous"], state["Hy_previous"], state["d_Ez_y"], state["d_Ez_x"],
            state["Psi_Bx_y"], state["Psi_By_x"], state["b_Bx_y"], state["c_Bx_y"],
            state["b_By_x"], state["c_By_x"], state["kappa_y_Hx"], state["kappa_x_Hy"],
            state["CaHx"], state["CbHx"], state["CaHy"], state["CbHy"],
            state["MRxx_Hx"], state["MRyy_Hy"], state["PMC_Hx"], state["PMC_Hy"])
        _tm_curl_h[blocks, THREADS_2D](
            state["Hx"], state["Hy"], state["d_Hx_y"], state["d_Hy_x"],
            sim.dx, sim.dy, periodic_x, periodic_y)
        _event_launch(h_events, state["d_Hx_y"], state["d_Hy_x"], step, sim.dt, source_device)
        _table_event_launch(bh_events, state["d_Hx_y"], state["d_Hy_x"], step)
        _tm_update_e[blocks, THREADS_2D](
            state["Ez"], state["Dz"], state["d_Hx_y"], state["d_Hy_x"],
            state["Psi_Dz_x"], state["Psi_Dz_y"], state["b_Dz_x"], state["c_Dz_x"],
            state["b_Dz_y"], state["c_Dz_y"], state["kappa_x_Ez"], state["kappa_y_Ez"],
            state["CaEz"], state["CbEz"], state["ERzz_Ez"])
        _event_launch(soft_events, state["Dz"], state["Dz"], step, sim.dt, source_device)
        _tm_finalize_e[blocks, THREADS_2D](state["Ez"], state["Dz"], state["ERzz_Ez"], state["PEC_Ez"])
        if monitor_blocks:
            _tm_sample_monitors[monitor_blocks, THREADS_1D](
                state["Ez"], state["Hx"], state["Hy"], state["Hx_previous"], state["Hy_previous"],
                monitors["x"], monitors["y"], monitors["offset"], monitors["stride"],
                monitors["it0"], monitors["it1"],
                monitors["values"], monitors["point_count"], step)
        if histories is not None and step % stride == 0:
            _tm_record_history[blocks, THREADS_2D](
                state["Hx"], state["Hy"], state["Ez"], state["Dz"],
                histories["Hx"], histories["Hy"], histories["Ez"], histories["Dz"], step // stride)
    cuda.synchronize()
    mutable = ("Ez", "Dz", "Hx", "Hy", "Bx", "By", "d_Ez_y", "d_Ez_x",
               "d_Hx_y", "d_Hy_x", "Psi_Bx_y", "Psi_By_x", "Psi_Dz_x", "Psi_Dz_y")
    _copy_back(sim, state, mutable)
    sim.Nt_rec = record_count
    if histories is not None:
        sim.Hx_history = histories["Hx"].copy_to_host()
        sim.Hy_history = histories["Hy"].copy_to_host()
        sim.Ez_history = histories["Ez"].copy_to_host()
        sim.Dz_history = histories["Dz"].copy_to_host()
    _finish_monitors(sim, monitors, ("Ez", "Hx", "Hy"))
    sim._gpu_state = None
    return sim.monitor_results


def run_te(sim, record_stride=1, is_include_history=True):
    """Run a TEz solver with all per-step state resident on the GPU."""
    sim._init_Coeff()
    stride = int(record_stride)
    if stride < 1:
        raise ValueError("record_stride must be positive.")
    sim.record_stride = stride
    sim.is_include_history = bool(is_include_history)
    state_names = (
        "Ex", "Ey", "Dx", "Dy", "Hz", "Bz", "d_Ex_y", "d_Ey_x",
        "d_Hz_y", "d_Hz_x", "Psi_Bz_x", "Psi_Bz_y", "Psi_Dx_y", "Psi_Dy_x",
        "b_Bz_x", "c_Bz_x", "b_Bz_y", "c_Bz_y", "b_Dx_y", "c_Dx_y",
        "b_Dy_x", "c_Dy_x", "kappa_x", "kappa_y", "kappa_y_Ex", "kappa_x_Ey",
        "CaHz", "CbHz", "CaEx", "CbEx", "CaEy", "CbEy", "MRzz_Hz",
        "ERxx_Ex", "ERyy_Ey", "PMC_Hz", "PEC_Ex", "PEC_Ey",
    )
    state = {name: _to_device(getattr(sim, name)) for name in state_names}
    state["Hz_previous"] = _to_device(sim.Hz)
    source_device = _source_device_arrays(sim)
    e_host, h_host, be_host, bh_host, soft_host = _compile_te_events(sim)
    e_events, h_events, soft_events = map(_device_events, (e_host, h_host, soft_host))
    be_events, bh_events = map(_device_table_events, (be_host, bh_host))
    monitors = _compile_monitors(sim)
    record_count = (sim.Nt + stride - 1) // stride if is_include_history else 0
    if is_include_history:
        histories = {
            "Ex": cuda.device_array((record_count,) + sim.Ex.shape, dtype=sim.Ex.dtype),
            "Ey": cuda.device_array((record_count,) + sim.Ey.shape, dtype=sim.Ey.dtype),
            "Hz": cuda.device_array((record_count,) + sim.Hz.shape, dtype=sim.Hz.dtype),
        }
    else:
        histories = None
    sim._gpu_state = state
    sim._gpu_transfer_stats = {
        "host_to_device_during_steps": 0,
        "device_to_host_during_steps": 0,
        "source_events": (
            e_host.count + h_host.count + be_host.count + bh_host.count + soft_host.count),
        "monitor_points": monitors["point_count"],
    }
    blocks = _blocks_2d(sim)
    periodic_x = "x" in getattr(sim, "periodic", "")
    periodic_y = "y" in getattr(sim, "periodic", "")
    monitor_blocks = ((monitors["point_count"] + THREADS_1D - 1) // THREADS_1D
                      if monitors["point_count"] else 0)
    for step in tqdm(range(sim.Nt), desc="FDTD simulation (GPU resident)", unit="step"):
        _te_curl_e[blocks, THREADS_2D](
            state["Ex"], state["Ey"], state["d_Ex_y"], state["d_Ey_x"], sim.dx, sim.dy)
        _event_launch(e_events, state["d_Ex_y"], state["d_Ey_x"], step, sim.dt, source_device)
        _table_event_launch(be_events, state["d_Ex_y"], state["d_Ey_x"], step)
        _te_update_h[blocks, THREADS_2D](
            state["Hz"], state["Bz"], state["Hz_previous"], state["d_Ex_y"], state["d_Ey_x"],
            state["Psi_Bz_x"], state["Psi_Bz_y"], state["b_Bz_x"], state["c_Bz_x"],
            state["b_Bz_y"], state["c_Bz_y"], state["kappa_x"], state["kappa_y"],
            state["CaHz"], state["CbHz"], state["MRzz_Hz"])
        _event_launch(soft_events, state["Bz"], state["Bz"], step, sim.dt, source_device)
        _te_finalize_h[blocks, THREADS_2D](state["Hz"], state["Bz"], state["MRzz_Hz"], state["PMC_Hz"])
        _te_curl_h[blocks, THREADS_2D](
            state["Hz"], state["d_Hz_y"], state["d_Hz_x"], sim.dx, sim.dy,
            periodic_x, periodic_y)
        _event_launch(h_events, state["d_Hz_y"], state["d_Hz_x"], step, sim.dt, source_device)
        _table_event_launch(bh_events, state["d_Hz_y"], state["d_Hz_x"], step)
        _te_update_e[blocks, THREADS_2D](
            state["Ex"], state["Ey"], state["Dx"], state["Dy"],
            state["d_Hz_y"], state["d_Hz_x"], state["Psi_Dx_y"], state["Psi_Dy_x"],
            state["b_Dx_y"], state["c_Dx_y"], state["b_Dy_x"], state["c_Dy_x"],
            state["kappa_y_Ex"], state["kappa_x_Ey"], state["CaEx"], state["CbEx"],
            state["CaEy"], state["CbEy"], state["ERxx_Ex"], state["ERyy_Ey"],
            state["PEC_Ex"], state["PEC_Ey"])
        if monitor_blocks:
            _te_sample_monitors[monitor_blocks, THREADS_1D](
                state["Ex"], state["Ey"], state["Hz"], state["Hz_previous"],
                monitors["x"], monitors["y"], monitors["offset"], monitors["stride"],
                monitors["it0"], monitors["it1"],
                monitors["values"], monitors["point_count"], step)
        if histories is not None and step % stride == 0:
            _te_record_history[blocks, THREADS_2D](
                state["Ex"], state["Ey"], state["Hz"], histories["Ex"],
                histories["Ey"], histories["Hz"], step // stride)
    cuda.synchronize()
    mutable = ("Ex", "Ey", "Dx", "Dy", "Hz", "Bz", "d_Ex_y", "d_Ey_x",
               "d_Hz_y", "d_Hz_x", "Psi_Bz_x", "Psi_Bz_y", "Psi_Dx_y", "Psi_Dy_x")
    _copy_back(sim, state, mutable)
    sim.Nt_rec = record_count
    if histories is not None:
        sim.Ex_history = histories["Ex"].copy_to_host()
        sim.Ey_history = histories["Ey"].copy_to_host()
        sim.Hz_history = histories["Hz"].copy_to_host()
    _finish_monitors(sim, monitors, ("Hz", "Ex", "Ey"))
    sim._gpu_state = None
    return sim.monitor_results
