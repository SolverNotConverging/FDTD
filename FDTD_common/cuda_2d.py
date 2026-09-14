"""Numerical CUDA stages. Allocation and graph setup live in runtime_2d."""

from __future__ import annotations

import math

from numba import cuda


THREADS_1D = 128
THREADS_2D = (32, 4)


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
                   shifts, count, clock, dt, modes, amplitudes, t0, tw,
                   frequencies):
    event = cuda.grid(1)
    if event >= count:
        return
    step = clock[0]
    value = factors[event] * _waveform(
        source_ids[event], step * dt + shifts[event], dt,
        modes, amplitudes, t0, tw, frequencies)
    index = (ii[event], jj[event])
    if targets[event] == 0:
        cuda.atomic.add(first, index, value)
    else:
        cuda.atomic.add(second, index, value)


@cuda.jit
def _inject_table_events(first, second, targets, ii, jj, values, count, clock):
    event = cuda.grid(1)
    if event >= count:
        return
    step = clock[0]
    index = (ii[event], jj[event])
    value = values[event, step]
    if targets[event] == 0:
        cuda.atomic.add(first, index, value)
    else:
        cuda.atomic.add(second, index, value)


@cuda.jit
def _tm_curl_e(ez, d_ez_y, d_ez_x, dx, dy):
    j, i = cuda.grid(2)  # x threads traverse contiguous columns
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
    j, i = cuda.grid(2)  # x threads traverse contiguous columns
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
    j, i = cuda.grid(2)  # x threads traverse contiguous columns
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
                 kappa_x_ez, kappa_y_ez, ca_ez, cb_ez, er_ez,
                 q, curl_scale, dispersive):
    j, i = cuda.grid(2)  # x threads traverse contiguous columns
    if i < ez.shape[0] and j < ez.shape[1]:
        psi_x = b_dz_x[i, j] * psi_dz_x[i, j] + c_dz_x[i, j] * d_hy_x[i, j]
        psi_y = b_dz_y[i, j] * psi_dz_y[i, j] + c_dz_y[i, j] * d_hx_y[i, j]
        psi_dz_x[i, j] = psi_x
        psi_dz_y[i, j] = psi_y
        curl = (
            d_hy_x[i, j] / kappa_x_ez[i, j]
            - d_hx_y[i, j] / kappa_y_ez[i, j] + psi_x - psi_y)
        if dispersive:
            trial = er_ez[i, j] * ez[i, j]
            for pole in range(q.shape[0]):
                trial += q[pole, i, j]
            dz[i, j] = trial + curl_scale * curl
        else:
            value = ca_ez[i, j] * ez[i, j] + cb_ez[i, j] * curl
            ez[i, j] = value
            dz[i, j] = er_ez[i, j] * value


@cuda.jit
def _finalize_e(ez, dz, er_ez, pec_ez, sigma, denominator,
                coefficients, r, q, v, dt, dispersive):
    j, i = cuda.grid(2)  # x threads traverse contiguous columns
    if i < ez.shape[0] and j < ez.shape[1]:
        if not dispersive:
            if pec_ez[i, j]:
                dz[i, j] = 0.0
            ez[i, j] = dz[i, j] / er_ez[i, j]
        else:
            old_e = ez[i, j]
            rhs = dz[i, j] - sigma[i, j] * old_e
            for pole in range(q.shape[0]):
                rhs -= (coefficients[pole, 0] * q[pole, i, j]
                        + coefficients[pole, 1] * v[pole, i, j]
                        + r[pole, i, j] * old_e)
            new_e = rhs / denominator[i, j]
            if pec_ez[i, j]:
                new_e = 0.0
            ez[i, j] = new_e
            displacement = er_ez[i, j] * new_e
            for pole in range(q.shape[0]):
                old_q = q[pole, i, j]
                old_v = v[pole, i, j]
                new_q = (coefficients[pole, 0] * old_q
                         + coefficients[pole, 1] * old_v
                         + r[pole, i, j] * (new_e + old_e))
                new_v = 0.0
                if coefficients[pole, 2] != 0.0:
                    new_v = 2.0 * (new_q - old_q) / dt - old_v
                if pec_ez[i, j]:
                    new_q = 0.0
                    new_v = 0.0
                q[pole, i, j] = new_q
                v[pole, i, j] = new_v
                displacement += new_q
            dz[i, j] = displacement


@cuda.jit
def _tm_sample_monitors(ez, hx, hy, hx_previous, hy_previous,
                        point_x, point_y, point_offset, point_stride,
                        point_it0, point_it1,
                        monitor_values, point_count, clock):
    point = cuda.grid(1)
    if point >= point_count:
        return
    step = clock[0]
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
                       ez_history, dz_history, clock, stride):
    j, i = cuda.grid(2)  # x threads traverse contiguous columns
    if clock[0] % stride != 0:
        return
    record_index = clock[0] // stride
    if record_index >= ez_history.shape[0]:
        return
    if i < hx.shape[0] and j < hx.shape[1]:
        hx_history[record_index, i, j] = hx[i, j]
    if i < hy.shape[0] and j < hy.shape[1]:
        hy_history[record_index, i, j] = hy[i, j]
    if i < ez.shape[0] and j < ez.shape[1]:
        ez_history[record_index, i, j] = ez[i, j]
        dz_history[record_index, i, j] = dz[i, j]


@cuda.jit
def _te_curl_e(ex, ey, d_ex_y, d_ey_x, dx, dy):
    j, i = cuda.grid(2)  # x threads traverse contiguous columns
    if i < d_ex_y.shape[0] and j < d_ex_y.shape[1]:
        d_ex_y[i, j] = (ex[i, j + 1] - ex[i, j]) / dy
        d_ey_x[i, j] = (ey[i + 1, j] - ey[i, j]) / dx


@cuda.jit
def _te_update_h(hz, bz, hz_previous, d_ex_y, d_ey_x,
                 psi_bz_x, psi_bz_y, b_bz_x, c_bz_x, b_bz_y, c_bz_y,
                 kappa_x, kappa_y, ca_hz, cb_hz, mr_hz):
    j, i = cuda.grid(2)  # x threads traverse contiguous columns
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
    j, i = cuda.grid(2)  # x threads traverse contiguous columns
    if i < hz.shape[0] and j < hz.shape[1]:
        if pmc_hz[i, j]:
            bz[i, j] = 0.0
            hz[i, j] = 0.0
        else:
            hz[i, j] = bz[i, j] / mr_hz[i, j]


@cuda.jit
def _te_curl_h(hz, d_hz_y, d_hz_x, dx, dy, periodic_x, periodic_y):
    j, i = cuda.grid(2)  # x threads traverse contiguous columns
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
                 er_ex, er_ey, pec_ex, pec_ey,
                 qx, qy, curl_scale, dispersive):
    j, i = cuda.grid(2)  # x threads traverse contiguous columns
    if i < ex.shape[0] and j < ex.shape[1]:
        psi = b_dx_y[i, j] * psi_dx_y[i, j] + c_dx_y[i, j] * d_hz_y[i, j]
        psi_dx_y[i, j] = psi
        curl = d_hz_y[i, j] / kappa_y_ex[i, j] + psi
        if dispersive:
            trial = er_ex[i, j] * ex[i, j]
            for pole in range(qx.shape[0]):
                trial += qx[pole, i, j]
            dx_field[i, j] = trial + curl_scale * curl
        else:
            value = ca_ex[i, j] * ex[i, j] + cb_ex[i, j] * curl
            if pec_ex[i, j]:
                value = 0.0
            dx_field[i, j] = er_ex[i, j] * value
            ex[i, j] = dx_field[i, j] / er_ex[i, j]
    if i < ey.shape[0] and j < ey.shape[1]:
        psi = b_dy_x[i, j] * psi_dy_x[i, j] + c_dy_x[i, j] * d_hz_x[i, j]
        psi_dy_x[i, j] = psi
        curl = d_hz_x[i, j] / kappa_x_ey[i, j] + psi
        if dispersive:
            trial = er_ey[i, j] * ey[i, j]
            for pole in range(qy.shape[0]):
                trial += qy[pole, i, j]
            dy_field[i, j] = trial - curl_scale * curl
        else:
            value = ca_ey[i, j] * ey[i, j] - cb_ey[i, j] * curl
            if pec_ey[i, j]:
                value = 0.0
            dy_field[i, j] = er_ey[i, j] * value
            ey[i, j] = dy_field[i, j] / er_ey[i, j]


@cuda.jit
def _te_sample_monitors(ex, ey, hz, hz_previous,
                        point_x, point_y, point_offset, point_stride,
                        point_it0, point_it1,
                        monitor_values, point_count, clock):
    point = cuda.grid(1)
    if point >= point_count:
        return
    step = clock[0]
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
                       clock, stride):
    j, i = cuda.grid(2)  # x threads traverse contiguous columns
    if clock[0] % stride != 0:
        return
    record_index = clock[0] // stride
    if record_index >= hz_history.shape[0]:
        return
    if i < ex.shape[0] and j < ex.shape[1]:
        ex_history[record_index, i, j] = ex[i, j]
    if i < ey.shape[0] and j < ey.shape[1]:
        ey_history[record_index, i, j] = ey[i, j]
    if i < hz.shape[0] and j < hz.shape[1]:
        hz_history[record_index, i, j] = hz[i, j]


@cuda.jit
def _advance_clock(clock):
    if cuda.grid(1) == 0:
        clock[0] += 1
