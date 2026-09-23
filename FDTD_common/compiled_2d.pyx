# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Native full-step loops. CUDA equivalents live in cuda_2d.py; parity tests cover both."""
from libc.math cimport exp, sin
from libc.stdint cimport int64_t

include "progress_2d.pxi"

cdef double _waveform(
    Py_ssize_t source_id, double time, double dt, int64_t[::1] modes, double[::1] amplitudes,
    double[::1] t0, double[::1] tw, double[::1] frequencies) noexcept nogil:
    cdef double amplitude
    cdef double center
    cdef double elapsed
    cdef double frequency
    cdef Py_ssize_t mode
    cdef double period
    cdef double ramp
    cdef double ramp_time
    cdef double ratio
    cdef double relative_time
    cdef double width
    mode = modes[source_id]
    amplitude = amplitudes[source_id]
    center = t0[source_id]
    width = tw[source_id]
    frequency = frequencies[source_id]
    relative_time = time - center
    if mode == 0:
        ratio = relative_time / width
        return amplitude * exp(-(ratio * ratio))
    if mode == 1:
        period = 1.0 / max(frequency, 1e-30)
        ramp_time = max(period, dt)
        elapsed = max(relative_time, 0.0)
        ramp = 1.0 - exp(-(elapsed / ramp_time) ** 3)
        return amplitude * ramp * sin(2.0 * 3.141592653589793 * frequency * relative_time)
    ratio = relative_time / width
    return amplitude * sin(2.0 * 3.141592653589793 * frequency * relative_time) * exp(-(ratio * ratio))

cdef void _inject_events(
    double[:, ::1] first, double[:, ::1] second, int64_t[::1] targets, int64_t[::1] ii, int64_t[::1] jj,
    int64_t[::1] source_ids, double[::1] factors, double[::1] shifts, Py_ssize_t count, int64_t[::1] clock,
    double dt, int64_t[::1] modes, double[::1] amplitudes, double[::1] t0, double[::1] tw,
    double[::1] frequencies) noexcept nogil:
    cdef Py_ssize_t event
    cdef Py_ssize_t step
    cdef double value
    for event in range(count):
        if event >= count:
            continue
        step = clock[0]
        value = factors[event] * _waveform(source_ids[event], step * dt + shifts[event], dt, modes, amplitudes, t0, tw, frequencies)
        if targets[event] == 0:
            first[ii[event], jj[event]] += value
        else:
            second[ii[event], jj[event]] += value

cdef void _inject_table_events(
    double[:, ::1] first, double[:, ::1] second, int64_t[::1] targets, int64_t[::1] ii, int64_t[::1] jj,
    double[:, ::1] values, Py_ssize_t count, int64_t[::1] clock) noexcept nogil:
    cdef Py_ssize_t event
    cdef Py_ssize_t step
    cdef double value
    for event in range(count):
        if event >= count:
            continue
        step = clock[0]
        value = values[event, step]
        if targets[event] == 0:
            first[ii[event], jj[event]] += value
        else:
            second[ii[event], jj[event]] += value

cdef void _tm_curl_e(
    double[:, ::1] ez, double[:, ::1] d_ez_y, double[:, ::1] d_ez_x, double dx, double dy,
    Py_ssize_t grid_nx, Py_ssize_t grid_ny) noexcept nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    for i in range(grid_nx + 1):
        for j in range(grid_ny + 1):
            if i < d_ez_y.shape[0] and j < d_ez_y.shape[1]:
                d_ez_y[i, j] = (ez[i, j + 1] - ez[i, j]) / dy
            if i < d_ez_x.shape[0] and j < d_ez_x.shape[1]:
                d_ez_x[i, j] = (ez[i + 1, j] - ez[i, j]) / dx

cdef void _tm_update_h(
    double[:, ::1] hx, double[:, ::1] hy, double[:, ::1] bx, double[:, ::1] by, double[:, ::1] hx_previous,
    double[:, ::1] hy_previous, double[:, ::1] d_ez_y, double[:, ::1] d_ez_x, double[:, ::1] psi_bx_y,
    double[:, ::1] psi_by_x, double[:, ::1] b_bx_y, double[:, ::1] c_bx_y, double[:, ::1] b_by_x,
    double[:, ::1] c_by_x, double[:, ::1] kappa_y_hx, double[:, ::1] kappa_x_hy, double[:, ::1] ca_hx,
    double[:, ::1] cb_hx, double[:, ::1] ca_hy, double[:, ::1] cb_hy, double[:, ::1] mr_hx,
    double[:, ::1] mr_hy, unsigned char[:, ::1] pmc_hx, unsigned char[:, ::1] pmc_hy, Py_ssize_t grid_nx,
    Py_ssize_t grid_ny) noexcept nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef double old
    cdef double psi
    cdef double value
    for i in range(grid_nx + 1):
        for j in range(grid_ny + 1):
            if i < hx.shape[0] and j < hx.shape[1]:
                old = hx[i, j]
                hx_previous[i, j] = old
                psi = b_bx_y[i, j] * psi_bx_y[i, j] + c_bx_y[i, j] * d_ez_y[i, j]
                psi_bx_y[i, j] = psi
                value = ca_hx[i, j] * old - cb_hx[i, j] * (d_ez_y[i, j] / kappa_y_hx[i, j] + psi)
                if pmc_hx[i, j]:
                    value = 0.0
                hx[i, j] = value
                bx[i, j] = mr_hx[i, j] * value
            if i < hy.shape[0] and j < hy.shape[1]:
                old = hy[i, j]
                hy_previous[i, j] = old
                psi = b_by_x[i, j] * psi_by_x[i, j] + c_by_x[i, j] * d_ez_x[i, j]
                psi_by_x[i, j] = psi
                value = ca_hy[i, j] * old + cb_hy[i, j] * (d_ez_x[i, j] / kappa_x_hy[i, j] + psi)
                if pmc_hy[i, j]:
                    value = 0.0
                hy[i, j] = value
                by[i, j] = mr_hy[i, j] * value

cdef void _tm_curl_h(
    double[:, ::1] hx, double[:, ::1] hy, double[:, ::1] d_hx_y, double[:, ::1] d_hy_x, double dx,
    double dy, bint periodic_x, bint periodic_y, Py_ssize_t grid_nx, Py_ssize_t grid_ny) noexcept nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t nx
    cdef Py_ssize_t ny
    for i in range(grid_nx + 1):
        for j in range(grid_ny + 1):
            nx = d_hx_y.shape[0] - 1
            ny = d_hx_y.shape[1] - 1
            if i <= nx and j <= ny:
                if j == 0:
                    d_hx_y[i, j] = (hx[i, 0] - hx[i, ny - 1] if periodic_y else hx[i, 0]) / dy
                elif j == ny:
                    d_hx_y[i, j] = (hx[i, 0] - hx[i, ny - 1] if periodic_y else -hx[i, ny - 1]) / dy
                else:
                    d_hx_y[i, j] = (hx[i, j] - hx[i, j - 1]) / dy
                if i == 0:
                    d_hy_x[i, j] = (hy[0, j] - hy[nx - 1, j] if periodic_x else hy[0, j]) / dx
                elif i == nx:
                    d_hy_x[i, j] = (hy[0, j] - hy[nx - 1, j] if periodic_x else -hy[nx - 1, j]) / dx
                else:
                    d_hy_x[i, j] = (hy[i, j] - hy[i - 1, j]) / dx

cdef void _tm_update_e(
    double[:, ::1] ez, double[:, ::1] dz, double[:, ::1] d_hx_y, double[:, ::1] d_hy_x,
    double[:, ::1] psi_dz_x, double[:, ::1] psi_dz_y, double[:, ::1] b_dz_x, double[:, ::1] c_dz_x,
    double[:, ::1] b_dz_y, double[:, ::1] c_dz_y, double[:, ::1] kappa_x_ez, double[:, ::1] kappa_y_ez,
    double[:, ::1] ca_ez, double[:, ::1] cb_ez, double[:, ::1] er_ez, double[:, :, ::1] q,
    double curl_scale, bint dispersive, Py_ssize_t grid_nx, Py_ssize_t grid_ny) noexcept nogil:
    cdef double curl
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t pole
    cdef double psi_x
    cdef double psi_y
    cdef double trial
    cdef double value
    for i in range(grid_nx + 1):
        for j in range(grid_ny + 1):
            if i < ez.shape[0] and j < ez.shape[1]:
                psi_x = b_dz_x[i, j] * psi_dz_x[i, j] + c_dz_x[i, j] * d_hy_x[i, j]
                psi_y = b_dz_y[i, j] * psi_dz_y[i, j] + c_dz_y[i, j] * d_hx_y[i, j]
                psi_dz_x[i, j] = psi_x
                psi_dz_y[i, j] = psi_y
                curl = d_hy_x[i, j] / kappa_x_ez[i, j] - d_hx_y[i, j] / kappa_y_ez[i, j] + psi_x - psi_y
                if dispersive:
                    trial = er_ez[i, j] * ez[i, j]
                    for pole in range(q.shape[0]):
                        trial += q[pole, i, j]
                    dz[i, j] = trial + curl_scale * curl
                else:
                    value = ca_ez[i, j] * ez[i, j] + cb_ez[i, j] * curl
                    ez[i, j] = value
                    dz[i, j] = er_ez[i, j] * value

cdef void _finalize_e(
    double[:, ::1] ez, double[:, ::1] dz, double[:, ::1] er_ez, unsigned char[:, ::1] pec_ez,
    double[:, ::1] sigma, double[:, ::1] denominator, double[:, ::1] coefficients, double[:, :, ::1] r,
    double[:, :, ::1] q, double[:, :, ::1] v, double dt, bint dispersive, Py_ssize_t grid_nx,
    Py_ssize_t grid_ny) noexcept nogil:
    cdef double displacement
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef double new_e
    cdef double new_q
    cdef double new_v
    cdef double old_e
    cdef double old_q
    cdef double old_v
    cdef Py_ssize_t pole
    cdef double rhs
    for i in range(grid_nx + 1):
        for j in range(grid_ny + 1):
            if i < ez.shape[0] and j < ez.shape[1]:
                if not dispersive:
                    if pec_ez[i, j]:
                        dz[i, j] = 0.0
                    ez[i, j] = dz[i, j] / er_ez[i, j]
                else:
                    old_e = ez[i, j]
                    rhs = dz[i, j] - sigma[i, j] * old_e
                    for pole in range(q.shape[0]):
                        rhs -= coefficients[pole, 0] * q[pole, i, j] + coefficients[pole, 1] * v[pole, i, j] + r[pole, i, j] * old_e
                    new_e = rhs / denominator[i, j]
                    if pec_ez[i, j]:
                        new_e = 0.0
                    ez[i, j] = new_e
                    displacement = er_ez[i, j] * new_e
                    for pole in range(q.shape[0]):
                        old_q = q[pole, i, j]
                        old_v = v[pole, i, j]
                        new_q = coefficients[pole, 0] * old_q + coefficients[pole, 1] * old_v + r[pole, i, j] * (new_e + old_e)
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

cdef void _tm_sample_monitors(
    double[:, ::1] ez, double[:, ::1] hx, double[:, ::1] hy, double[:, ::1] hx_previous,
    double[:, ::1] hy_previous, int64_t[::1] point_x, int64_t[::1] point_y, int64_t[::1] point_offset,
    int64_t[::1] point_stride, int64_t[::1] point_it0, int64_t[::1] point_it1,
    double[:, ::1] monitor_values, Py_ssize_t point_count, int64_t[::1] clock) noexcept nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t it0
    cdef Py_ssize_t j
    cdef Py_ssize_t output
    cdef Py_ssize_t point
    cdef Py_ssize_t step
    for point in range(point_count):
        if point >= point_count:
            continue
        step = clock[0]
        it0 = point_it0[point]
        if step < it0 or step >= point_it1[point]:
            continue
        i = point_x[point]
        j = point_y[point]
        output = point_offset[point] + (step - it0) * point_stride[point]
        monitor_values[output, 0] = 0.25 * (ez[i, j] + ez[i + 1, j] + ez[i, j + 1] + ez[i + 1, j + 1])
        monitor_values[output, 1] = 0.25 * (hx[i, j] + hx_previous[i, j] + hx[i + 1, j] + hx_previous[i + 1, j])
        monitor_values[output, 2] = 0.25 * (hy[i, j] + hy_previous[i, j] + hy[i, j + 1] + hy_previous[i, j + 1])

cdef void _tm_record_history(
    double[:, ::1] hx, double[:, ::1] hy, double[:, ::1] ez, double[:, ::1] dz,
    double[:, :, ::1] hx_history, double[:, :, ::1] hy_history, double[:, :, ::1] ez_history,
    double[:, :, ::1] dz_history, int64_t[::1] clock, Py_ssize_t stride, Py_ssize_t grid_nx,
    Py_ssize_t grid_ny) noexcept nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t record_index
    if ez_history.shape[0] == 0 or clock[0] % stride != 0:
        return
    for i in range(grid_nx + 1):
        for j in range(grid_ny + 1):
            if clock[0] % stride != 0:
                continue
            record_index = clock[0] // stride
            if record_index >= ez_history.shape[0]:
                continue
            if i < hx.shape[0] and j < hx.shape[1]:
                hx_history[record_index, i, j] = hx[i, j]
            if i < hy.shape[0] and j < hy.shape[1]:
                hy_history[record_index, i, j] = hy[i, j]
            if i < ez.shape[0] and j < ez.shape[1]:
                ez_history[record_index, i, j] = ez[i, j]
                dz_history[record_index, i, j] = dz[i, j]

cdef void _te_curl_e(
    double[:, ::1] ex, double[:, ::1] ey, double[:, ::1] d_ex_y, double[:, ::1] d_ey_x, double dx,
    double dy, Py_ssize_t grid_nx, Py_ssize_t grid_ny) noexcept nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    for i in range(grid_nx + 1):
        for j in range(grid_ny + 1):
            if i < d_ex_y.shape[0] and j < d_ex_y.shape[1]:
                d_ex_y[i, j] = (ex[i, j + 1] - ex[i, j]) / dy
                d_ey_x[i, j] = (ey[i + 1, j] - ey[i, j]) / dx

cdef void _te_update_h(
    double[:, ::1] hz, double[:, ::1] bz, double[:, ::1] hz_previous, double[:, ::1] d_ex_y,
    double[:, ::1] d_ey_x, double[:, ::1] psi_bz_x, double[:, ::1] psi_bz_y, double[:, ::1] b_bz_x,
    double[:, ::1] c_bz_x, double[:, ::1] b_bz_y, double[:, ::1] c_bz_y, double[:, ::1] kappa_x,
    double[:, ::1] kappa_y, double[:, ::1] ca_hz, double[:, ::1] cb_hz, double[:, ::1] mr_hz,
    Py_ssize_t grid_nx, Py_ssize_t grid_ny) noexcept nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef double old
    cdef double psi_x
    cdef double psi_y
    cdef double value
    for i in range(grid_nx + 1):
        for j in range(grid_ny + 1):
            if i < hz.shape[0] and j < hz.shape[1]:
                old = hz[i, j]
                hz_previous[i, j] = old
                psi_x = b_bz_x[i, j] * psi_bz_x[i, j] + c_bz_x[i, j] * d_ey_x[i, j]
                psi_y = b_bz_y[i, j] * psi_bz_y[i, j] + c_bz_y[i, j] * d_ex_y[i, j]
                psi_bz_x[i, j] = psi_x
                psi_bz_y[i, j] = psi_y
                value = ca_hz[i, j] * old - cb_hz[i, j] * (d_ey_x[i, j] / kappa_x[i, j] - d_ex_y[i, j] / kappa_y[i, j] + psi_x - psi_y)
                hz[i, j] = value
                bz[i, j] = mr_hz[i, j] * value

cdef void _te_finalize_h(
    double[:, ::1] hz, double[:, ::1] bz, double[:, ::1] mr_hz, unsigned char[:, ::1] pmc_hz,
    Py_ssize_t grid_nx, Py_ssize_t grid_ny) noexcept nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    for i in range(grid_nx + 1):
        for j in range(grid_ny + 1):
            if i < hz.shape[0] and j < hz.shape[1]:
                if pmc_hz[i, j]:
                    bz[i, j] = 0.0
                    hz[i, j] = 0.0
                else:
                    hz[i, j] = bz[i, j] / mr_hz[i, j]

cdef void _te_curl_h(
    double[:, ::1] hz, double[:, ::1] d_hz_y, double[:, ::1] d_hz_x, double dx, double dy, bint periodic_x,
    bint periodic_y, Py_ssize_t grid_nx, Py_ssize_t grid_ny) noexcept nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t nx
    cdef Py_ssize_t ny
    for i in range(grid_nx + 1):
        for j in range(grid_ny + 1):
            nx = hz.shape[0]
            ny = hz.shape[1]
            if i < nx and j <= ny:
                if j == 0:
                    d_hz_y[i, j] = (hz[i, 0] - hz[i, ny - 1] if periodic_y else hz[i, 0]) / dy
                elif j == ny:
                    d_hz_y[i, j] = (hz[i, 0] - hz[i, ny - 1] if periodic_y else -hz[i, ny - 1]) / dy
                else:
                    d_hz_y[i, j] = (hz[i, j] - hz[i, j - 1]) / dy
            if i <= nx and j < ny:
                if i == 0:
                    d_hz_x[i, j] = (hz[0, j] - hz[nx - 1, j] if periodic_x else hz[0, j]) / dx
                elif i == nx:
                    d_hz_x[i, j] = (hz[0, j] - hz[nx - 1, j] if periodic_x else -hz[nx - 1, j]) / dx
                else:
                    d_hz_x[i, j] = (hz[i, j] - hz[i - 1, j]) / dx

cdef void _te_update_e(
    double[:, ::1] ex, double[:, ::1] ey, double[:, ::1] dx_field, double[:, ::1] dy_field,
    double[:, ::1] d_hz_y, double[:, ::1] d_hz_x, double[:, ::1] psi_dx_y, double[:, ::1] psi_dy_x,
    double[:, ::1] b_dx_y, double[:, ::1] c_dx_y, double[:, ::1] b_dy_x, double[:, ::1] c_dy_x,
    double[:, ::1] kappa_y_ex, double[:, ::1] kappa_x_ey, double[:, ::1] ca_ex, double[:, ::1] cb_ex,
    double[:, ::1] ca_ey, double[:, ::1] cb_ey, double[:, ::1] er_ex, double[:, ::1] er_ey,
    unsigned char[:, ::1] pec_ex, unsigned char[:, ::1] pec_ey, double[:, :, ::1] qx, double[:, :, ::1] qy,
    double curl_scale, bint dispersive, Py_ssize_t grid_nx, Py_ssize_t grid_ny) noexcept nogil:
    cdef double curl
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t pole
    cdef double psi
    cdef double trial
    cdef double value
    for i in range(grid_nx + 1):
        for j in range(grid_ny + 1):
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

cdef void _te_sample_monitors(
    double[:, ::1] ex, double[:, ::1] ey, double[:, ::1] hz, double[:, ::1] hz_previous,
    int64_t[::1] point_x, int64_t[::1] point_y, int64_t[::1] point_offset, int64_t[::1] point_stride,
    int64_t[::1] point_it0, int64_t[::1] point_it1, double[:, ::1] monitor_values, Py_ssize_t point_count,
    int64_t[::1] clock) noexcept nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t it0
    cdef Py_ssize_t j
    cdef Py_ssize_t output
    cdef Py_ssize_t point
    cdef Py_ssize_t step
    for point in range(point_count):
        if point >= point_count:
            continue
        step = clock[0]
        it0 = point_it0[point]
        if step < it0 or step >= point_it1[point]:
            continue
        i = point_x[point]
        j = point_y[point]
        output = point_offset[point] + (step - it0) * point_stride[point]
        monitor_values[output, 0] = 0.5 * (hz[i, j] + hz_previous[i, j])
        monitor_values[output, 1] = 0.5 * (ex[i, j] + ex[i, j + 1])
        monitor_values[output, 2] = 0.5 * (ey[i, j] + ey[i + 1, j])

cdef void _te_record_history(
    double[:, ::1] ex, double[:, ::1] ey, double[:, ::1] hz, double[:, :, ::1] ex_history,
    double[:, :, ::1] ey_history, double[:, :, ::1] hz_history, int64_t[::1] clock, Py_ssize_t stride,
    Py_ssize_t grid_nx, Py_ssize_t grid_ny) noexcept nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t record_index
    if hz_history.shape[0] == 0 or clock[0] % stride != 0:
        return
    for i in range(grid_nx + 1):
        for j in range(grid_ny + 1):
            if clock[0] % stride != 0:
                continue
            record_index = clock[0] // stride
            if record_index >= hz_history.shape[0]:
                continue
            if i < ex.shape[0] and j < ex.shape[1]:
                ex_history[record_index, i, j] = ex[i, j]
            if i < ey.shape[0] and j < ey.shape[1]:
                ey_history[record_index, i, j] = ey[i, j]
            if i < hz.shape[0] and j < hz.shape[1]:
                hz_history[record_index, i, j] = hz[i, j]

cdef void _advance_clock(int64_t[::1] clock) noexcept nogil:
    clock[0] += 1

cdef void _cut_tm_curl_e(
    double[:, ::1] ez, double[:, ::1] d_ez_y, double[:, ::1] d_ez_x,
    double[:, ::1] open_hx, double[:, ::1] open_hy, double dx, double dy,
    Py_ssize_t nx, Py_ssize_t ny) noexcept nogil:
    cdef Py_ssize_t i, j
    for i in range(nx + 1):
        for j in range(ny + 1):
            if j < ny:
                d_ez_y[i, j] = ((ez[i, j + 1] - ez[i, j]) / (dy * open_hx[i, j])
                                 if open_hx[i, j] > 0 else 0.0)
            if i < nx:
                d_ez_x[i, j] = ((ez[i + 1, j] - ez[i, j]) / (dx * open_hy[i, j])
                                 if open_hy[i, j] > 0 else 0.0)

cdef void _cut_te_curl_e(
    double[:, ::1] ex, double[:, ::1] ey, double[:, ::1] d_ex_y,
    double[:, ::1] d_ey_x, double[:, ::1] area, double[:, ::1] open_ex,
    double[:, ::1] open_ey, double dx, double dy,
    Py_ssize_t nx, Py_ssize_t ny) noexcept nogil:
    cdef Py_ssize_t i, j
    for i in range(nx):
        for j in range(ny):
            if area[i, j] > 0:
                d_ex_y[i, j] = (ex[i, j + 1] * open_ex[i, j + 1]
                                - ex[i, j] * open_ex[i, j]) / (dy * area[i, j])
                d_ey_x[i, j] = (ey[i + 1, j] * open_ey[i + 1, j]
                                - ey[i, j] * open_ey[i, j]) / (dx * area[i, j])
            else:
                d_ex_y[i, j] = 0.0
                d_ey_x[i, j] = 0.0

cdef void _cut_merge_h(
    double[:, ::1] field, double[:, ::1] flux, double[:, ::1] material,
    int64_t[::1] offsets, int64_t[::1] ii, int64_t[::1] jj,
    double[::1] weights) noexcept nogil:
    cdef Py_ssize_t group, member, i, j
    cdef double value
    for group in range(offsets.shape[0] - 1):
        value = 0.0
        for member in range(offsets[group], offsets[group + 1]):
            value += weights[member] * field[ii[member], jj[member]]
        for member in range(offsets[group], offsets[group + 1]):
            i = ii[member]
            j = jj[member]
            field[i, j] = value
            flux[i, j] = material[i, j] * value

cdef void _cut_zero_h(
    double[:, ::1] field, double[:, ::1] flux, double[:, ::1] fraction) noexcept nogil:
    cdef Py_ssize_t i, j
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            if fraction[i, j] == 0:
                field[i, j] = 0.0
                flux[i, j] = 0.0

cpdef run_tm(tuple stages, Py_ssize_t steps, Py_ssize_t nx, Py_ssize_t ny,
             bint progress=False, tuple cutcell=()):
    if len(stages) != 13 or steps < 0 or nx < 1 or ny < 1:
        raise ValueError("Invalid prepared 2D execution plan.")
    cdef Py_ssize_t step
    cdef double[:, ::1] s0_ez = stages[0][0]
    cdef double[:, ::1] s0_d_ez_y = stages[0][1]
    cdef double[:, ::1] s0_d_ez_x = stages[0][2]
    cdef double s0_dx = stages[0][3]
    cdef double s0_dy = stages[0][4]
    cdef double[:, ::1] s1_first = stages[1][0]
    cdef double[:, ::1] s1_second = stages[1][1]
    cdef int64_t[::1] s1_targets = stages[1][2]
    cdef int64_t[::1] s1_ii = stages[1][3]
    cdef int64_t[::1] s1_jj = stages[1][4]
    cdef int64_t[::1] s1_source_ids = stages[1][5]
    cdef double[::1] s1_factors = stages[1][6]
    cdef double[::1] s1_shifts = stages[1][7]
    cdef Py_ssize_t s1_count = stages[1][8]
    cdef int64_t[::1] s1_clock = stages[1][9]
    cdef double s1_dt = stages[1][10]
    cdef int64_t[::1] s1_modes = stages[1][11]
    cdef double[::1] s1_amplitudes = stages[1][12]
    cdef double[::1] s1_t0 = stages[1][13]
    cdef double[::1] s1_tw = stages[1][14]
    cdef double[::1] s1_frequencies = stages[1][15]
    cdef double[:, ::1] s2_first = stages[2][0]
    cdef double[:, ::1] s2_second = stages[2][1]
    cdef int64_t[::1] s2_targets = stages[2][2]
    cdef int64_t[::1] s2_ii = stages[2][3]
    cdef int64_t[::1] s2_jj = stages[2][4]
    cdef double[:, ::1] s2_values = stages[2][5]
    cdef Py_ssize_t s2_count = stages[2][6]
    cdef int64_t[::1] s2_clock = stages[2][7]
    cdef double[:, ::1] s3_hx = stages[3][0]
    cdef double[:, ::1] s3_hy = stages[3][1]
    cdef double[:, ::1] s3_bx = stages[3][2]
    cdef double[:, ::1] s3_by = stages[3][3]
    cdef double[:, ::1] s3_hx_previous = stages[3][4]
    cdef double[:, ::1] s3_hy_previous = stages[3][5]
    cdef double[:, ::1] s3_d_ez_y = stages[3][6]
    cdef double[:, ::1] s3_d_ez_x = stages[3][7]
    cdef double[:, ::1] s3_psi_bx_y = stages[3][8]
    cdef double[:, ::1] s3_psi_by_x = stages[3][9]
    cdef double[:, ::1] s3_b_bx_y = stages[3][10]
    cdef double[:, ::1] s3_c_bx_y = stages[3][11]
    cdef double[:, ::1] s3_b_by_x = stages[3][12]
    cdef double[:, ::1] s3_c_by_x = stages[3][13]
    cdef double[:, ::1] s3_kappa_y_hx = stages[3][14]
    cdef double[:, ::1] s3_kappa_x_hy = stages[3][15]
    cdef double[:, ::1] s3_ca_hx = stages[3][16]
    cdef double[:, ::1] s3_cb_hx = stages[3][17]
    cdef double[:, ::1] s3_ca_hy = stages[3][18]
    cdef double[:, ::1] s3_cb_hy = stages[3][19]
    cdef double[:, ::1] s3_mr_hx = stages[3][20]
    cdef double[:, ::1] s3_mr_hy = stages[3][21]
    cdef unsigned char[:, ::1] s3_pmc_hx = stages[3][22]
    cdef unsigned char[:, ::1] s3_pmc_hy = stages[3][23]
    cdef double[:, ::1] s4_hx = stages[4][0]
    cdef double[:, ::1] s4_hy = stages[4][1]
    cdef double[:, ::1] s4_d_hx_y = stages[4][2]
    cdef double[:, ::1] s4_d_hy_x = stages[4][3]
    cdef double s4_dx = stages[4][4]
    cdef double s4_dy = stages[4][5]
    cdef bint s4_periodic_x = stages[4][6]
    cdef bint s4_periodic_y = stages[4][7]
    cdef double[:, ::1] s5_first = stages[5][0]
    cdef double[:, ::1] s5_second = stages[5][1]
    cdef int64_t[::1] s5_targets = stages[5][2]
    cdef int64_t[::1] s5_ii = stages[5][3]
    cdef int64_t[::1] s5_jj = stages[5][4]
    cdef int64_t[::1] s5_source_ids = stages[5][5]
    cdef double[::1] s5_factors = stages[5][6]
    cdef double[::1] s5_shifts = stages[5][7]
    cdef Py_ssize_t s5_count = stages[5][8]
    cdef int64_t[::1] s5_clock = stages[5][9]
    cdef double s5_dt = stages[5][10]
    cdef int64_t[::1] s5_modes = stages[5][11]
    cdef double[::1] s5_amplitudes = stages[5][12]
    cdef double[::1] s5_t0 = stages[5][13]
    cdef double[::1] s5_tw = stages[5][14]
    cdef double[::1] s5_frequencies = stages[5][15]
    cdef double[:, ::1] s6_first = stages[6][0]
    cdef double[:, ::1] s6_second = stages[6][1]
    cdef int64_t[::1] s6_targets = stages[6][2]
    cdef int64_t[::1] s6_ii = stages[6][3]
    cdef int64_t[::1] s6_jj = stages[6][4]
    cdef double[:, ::1] s6_values = stages[6][5]
    cdef Py_ssize_t s6_count = stages[6][6]
    cdef int64_t[::1] s6_clock = stages[6][7]
    cdef double[:, ::1] s7_ez = stages[7][0]
    cdef double[:, ::1] s7_dz = stages[7][1]
    cdef double[:, ::1] s7_d_hx_y = stages[7][2]
    cdef double[:, ::1] s7_d_hy_x = stages[7][3]
    cdef double[:, ::1] s7_psi_dz_x = stages[7][4]
    cdef double[:, ::1] s7_psi_dz_y = stages[7][5]
    cdef double[:, ::1] s7_b_dz_x = stages[7][6]
    cdef double[:, ::1] s7_c_dz_x = stages[7][7]
    cdef double[:, ::1] s7_b_dz_y = stages[7][8]
    cdef double[:, ::1] s7_c_dz_y = stages[7][9]
    cdef double[:, ::1] s7_kappa_x_ez = stages[7][10]
    cdef double[:, ::1] s7_kappa_y_ez = stages[7][11]
    cdef double[:, ::1] s7_ca_ez = stages[7][12]
    cdef double[:, ::1] s7_cb_ez = stages[7][13]
    cdef double[:, ::1] s7_er_ez = stages[7][14]
    cdef double[:, :, ::1] s7_q = stages[7][15]
    cdef double s7_curl_scale = stages[7][16]
    cdef bint s7_dispersive = stages[7][17]
    cdef double[:, ::1] s8_first = stages[8][0]
    cdef double[:, ::1] s8_second = stages[8][1]
    cdef int64_t[::1] s8_targets = stages[8][2]
    cdef int64_t[::1] s8_ii = stages[8][3]
    cdef int64_t[::1] s8_jj = stages[8][4]
    cdef int64_t[::1] s8_source_ids = stages[8][5]
    cdef double[::1] s8_factors = stages[8][6]
    cdef double[::1] s8_shifts = stages[8][7]
    cdef Py_ssize_t s8_count = stages[8][8]
    cdef int64_t[::1] s8_clock = stages[8][9]
    cdef double s8_dt = stages[8][10]
    cdef int64_t[::1] s8_modes = stages[8][11]
    cdef double[::1] s8_amplitudes = stages[8][12]
    cdef double[::1] s8_t0 = stages[8][13]
    cdef double[::1] s8_tw = stages[8][14]
    cdef double[::1] s8_frequencies = stages[8][15]
    cdef double[:, ::1] s9_ez = stages[9][0]
    cdef double[:, ::1] s9_dz = stages[9][1]
    cdef double[:, ::1] s9_er_ez = stages[9][2]
    cdef unsigned char[:, ::1] s9_pec_ez = stages[9][3]
    cdef double[:, ::1] s9_sigma = stages[9][4]
    cdef double[:, ::1] s9_denominator = stages[9][5]
    cdef double[:, ::1] s9_coefficients = stages[9][6]
    cdef double[:, :, ::1] s9_r = stages[9][7]
    cdef double[:, :, ::1] s9_q = stages[9][8]
    cdef double[:, :, ::1] s9_v = stages[9][9]
    cdef double s9_dt = stages[9][10]
    cdef bint s9_dispersive = stages[9][11]
    cdef double[:, ::1] s10_ez = stages[10][0]
    cdef double[:, ::1] s10_hx = stages[10][1]
    cdef double[:, ::1] s10_hy = stages[10][2]
    cdef double[:, ::1] s10_hx_previous = stages[10][3]
    cdef double[:, ::1] s10_hy_previous = stages[10][4]
    cdef int64_t[::1] s10_point_x = stages[10][5]
    cdef int64_t[::1] s10_point_y = stages[10][6]
    cdef int64_t[::1] s10_point_offset = stages[10][7]
    cdef int64_t[::1] s10_point_stride = stages[10][8]
    cdef int64_t[::1] s10_point_it0 = stages[10][9]
    cdef int64_t[::1] s10_point_it1 = stages[10][10]
    cdef double[:, ::1] s10_monitor_values = stages[10][11]
    cdef Py_ssize_t s10_point_count = stages[10][12]
    cdef int64_t[::1] s10_clock = stages[10][13]
    cdef double[:, ::1] s11_hx = stages[11][0]
    cdef double[:, ::1] s11_hy = stages[11][1]
    cdef double[:, ::1] s11_ez = stages[11][2]
    cdef double[:, ::1] s11_dz = stages[11][3]
    cdef double[:, :, ::1] s11_hx_history = stages[11][4]
    cdef double[:, :, ::1] s11_hy_history = stages[11][5]
    cdef double[:, :, ::1] s11_ez_history = stages[11][6]
    cdef double[:, :, ::1] s11_dz_history = stages[11][7]
    cdef int64_t[::1] s11_clock = stages[11][8]
    cdef Py_ssize_t s11_stride = stages[11][9]
    cdef int64_t[::1] s12_clock = stages[12][0]
    cdef bint has_cut = len(cutcell) != 0
    cdef double[:, ::1] cut_open_hx = cutcell[0] if has_cut else s3_mr_hx
    cdef double[:, ::1] cut_open_hy = cutcell[1] if has_cut else s3_mr_hy
    cdef int64_t[::1] cut_hx_offsets = cutcell[2] if has_cut else s1_ii
    cdef int64_t[::1] cut_hx_ii = cutcell[3] if has_cut else s1_ii
    cdef int64_t[::1] cut_hx_jj = cutcell[4] if has_cut else s1_jj
    cdef double[::1] cut_hx_weights = cutcell[5] if has_cut else s1_factors
    cdef int64_t[::1] cut_hy_offsets = cutcell[6] if has_cut else s1_ii
    cdef int64_t[::1] cut_hy_ii = cutcell[7] if has_cut else s1_ii
    cdef int64_t[::1] cut_hy_jj = cutcell[8] if has_cut else s1_jj
    cdef double[::1] cut_hy_weights = cutcell[9] if has_cut else s1_factors
    cdef NativeProgress reporter
    with nogil:
        if progress:
            progress_start(&reporter, steps)
        for step in range(steps):
            if has_cut:
                _cut_tm_curl_e(s0_ez, s0_d_ez_y, s0_d_ez_x,
                               cut_open_hx, cut_open_hy, s0_dx, s0_dy, nx, ny)
            else:
                _tm_curl_e(s0_ez, s0_d_ez_y, s0_d_ez_x, s0_dx, s0_dy, nx, ny)
            _inject_events(
                s1_first, s1_second, s1_targets, s1_ii, s1_jj, s1_source_ids, s1_factors, s1_shifts,
                s1_count, s1_clock, s1_dt, s1_modes, s1_amplitudes, s1_t0, s1_tw, s1_frequencies)
            _inject_table_events(s2_first, s2_second, s2_targets, s2_ii, s2_jj, s2_values, s2_count, s2_clock)
            _tm_update_h(
                s3_hx, s3_hy, s3_bx, s3_by, s3_hx_previous, s3_hy_previous, s3_d_ez_y, s3_d_ez_x,
                s3_psi_bx_y, s3_psi_by_x, s3_b_bx_y, s3_c_bx_y, s3_b_by_x, s3_c_by_x, s3_kappa_y_hx,
                s3_kappa_x_hy, s3_ca_hx, s3_cb_hx, s3_ca_hy, s3_cb_hy, s3_mr_hx, s3_mr_hy, s3_pmc_hx,
                s3_pmc_hy, nx, ny)
            if has_cut:
                _cut_merge_h(s3_hx, s3_bx, s3_mr_hx, cut_hx_offsets,
                             cut_hx_ii, cut_hx_jj, cut_hx_weights)
                _cut_merge_h(s3_hy, s3_by, s3_mr_hy, cut_hy_offsets,
                             cut_hy_ii, cut_hy_jj, cut_hy_weights)
                _cut_zero_h(s3_hx, s3_bx, cut_open_hx)
                _cut_zero_h(s3_hy, s3_by, cut_open_hy)
            _tm_curl_h(s4_hx, s4_hy, s4_d_hx_y, s4_d_hy_x, s4_dx, s4_dy, s4_periodic_x, s4_periodic_y, nx, ny)
            _inject_events(
                s5_first, s5_second, s5_targets, s5_ii, s5_jj, s5_source_ids, s5_factors, s5_shifts,
                s5_count, s5_clock, s5_dt, s5_modes, s5_amplitudes, s5_t0, s5_tw, s5_frequencies)
            _inject_table_events(s6_first, s6_second, s6_targets, s6_ii, s6_jj, s6_values, s6_count, s6_clock)
            _tm_update_e(
                s7_ez, s7_dz, s7_d_hx_y, s7_d_hy_x, s7_psi_dz_x, s7_psi_dz_y, s7_b_dz_x, s7_c_dz_x,
                s7_b_dz_y, s7_c_dz_y, s7_kappa_x_ez, s7_kappa_y_ez, s7_ca_ez, s7_cb_ez, s7_er_ez, s7_q,
                s7_curl_scale, s7_dispersive, nx, ny)
            _inject_events(
                s8_first, s8_second, s8_targets, s8_ii, s8_jj, s8_source_ids, s8_factors, s8_shifts,
                s8_count, s8_clock, s8_dt, s8_modes, s8_amplitudes, s8_t0, s8_tw, s8_frequencies)
            _finalize_e(
                s9_ez, s9_dz, s9_er_ez, s9_pec_ez, s9_sigma, s9_denominator, s9_coefficients, s9_r, s9_q,
                s9_v, s9_dt, s9_dispersive, nx, ny)
            _tm_sample_monitors(
                s10_ez, s10_hx, s10_hy, s10_hx_previous, s10_hy_previous, s10_point_x, s10_point_y,
                s10_point_offset, s10_point_stride, s10_point_it0, s10_point_it1, s10_monitor_values,
                s10_point_count, s10_clock)
            _tm_record_history(
                s11_hx, s11_hy, s11_ez, s11_dz, s11_hx_history, s11_hy_history, s11_ez_history,
                s11_dz_history, s11_clock, s11_stride, nx, ny)
            _advance_clock(s12_clock)
            if progress and (step + 1) % 64 == 0 and step + 1 < steps:
                progress_update(&reporter, step + 1, steps)
        if progress:
            progress_update(&reporter, steps, steps, True)

cpdef run_te(tuple stages, Py_ssize_t steps, Py_ssize_t nx, Py_ssize_t ny,
             bint progress=False, tuple cutcell=()):
    if len(stages) != 15 or steps < 0 or nx < 1 or ny < 1:
        raise ValueError("Invalid prepared 2D execution plan.")
    cdef Py_ssize_t step
    cdef double[:, ::1] s0_ex = stages[0][0]
    cdef double[:, ::1] s0_ey = stages[0][1]
    cdef double[:, ::1] s0_d_ex_y = stages[0][2]
    cdef double[:, ::1] s0_d_ey_x = stages[0][3]
    cdef double s0_dx = stages[0][4]
    cdef double s0_dy = stages[0][5]
    cdef double[:, ::1] s1_first = stages[1][0]
    cdef double[:, ::1] s1_second = stages[1][1]
    cdef int64_t[::1] s1_targets = stages[1][2]
    cdef int64_t[::1] s1_ii = stages[1][3]
    cdef int64_t[::1] s1_jj = stages[1][4]
    cdef int64_t[::1] s1_source_ids = stages[1][5]
    cdef double[::1] s1_factors = stages[1][6]
    cdef double[::1] s1_shifts = stages[1][7]
    cdef Py_ssize_t s1_count = stages[1][8]
    cdef int64_t[::1] s1_clock = stages[1][9]
    cdef double s1_dt = stages[1][10]
    cdef int64_t[::1] s1_modes = stages[1][11]
    cdef double[::1] s1_amplitudes = stages[1][12]
    cdef double[::1] s1_t0 = stages[1][13]
    cdef double[::1] s1_tw = stages[1][14]
    cdef double[::1] s1_frequencies = stages[1][15]
    cdef double[:, ::1] s2_first = stages[2][0]
    cdef double[:, ::1] s2_second = stages[2][1]
    cdef int64_t[::1] s2_targets = stages[2][2]
    cdef int64_t[::1] s2_ii = stages[2][3]
    cdef int64_t[::1] s2_jj = stages[2][4]
    cdef double[:, ::1] s2_values = stages[2][5]
    cdef Py_ssize_t s2_count = stages[2][6]
    cdef int64_t[::1] s2_clock = stages[2][7]
    cdef double[:, ::1] s3_hz = stages[3][0]
    cdef double[:, ::1] s3_bz = stages[3][1]
    cdef double[:, ::1] s3_hz_previous = stages[3][2]
    cdef double[:, ::1] s3_d_ex_y = stages[3][3]
    cdef double[:, ::1] s3_d_ey_x = stages[3][4]
    cdef double[:, ::1] s3_psi_bz_x = stages[3][5]
    cdef double[:, ::1] s3_psi_bz_y = stages[3][6]
    cdef double[:, ::1] s3_b_bz_x = stages[3][7]
    cdef double[:, ::1] s3_c_bz_x = stages[3][8]
    cdef double[:, ::1] s3_b_bz_y = stages[3][9]
    cdef double[:, ::1] s3_c_bz_y = stages[3][10]
    cdef double[:, ::1] s3_kappa_x = stages[3][11]
    cdef double[:, ::1] s3_kappa_y = stages[3][12]
    cdef double[:, ::1] s3_ca_hz = stages[3][13]
    cdef double[:, ::1] s3_cb_hz = stages[3][14]
    cdef double[:, ::1] s3_mr_hz = stages[3][15]
    cdef double[:, ::1] s4_first = stages[4][0]
    cdef double[:, ::1] s4_second = stages[4][1]
    cdef int64_t[::1] s4_targets = stages[4][2]
    cdef int64_t[::1] s4_ii = stages[4][3]
    cdef int64_t[::1] s4_jj = stages[4][4]
    cdef int64_t[::1] s4_source_ids = stages[4][5]
    cdef double[::1] s4_factors = stages[4][6]
    cdef double[::1] s4_shifts = stages[4][7]
    cdef Py_ssize_t s4_count = stages[4][8]
    cdef int64_t[::1] s4_clock = stages[4][9]
    cdef double s4_dt = stages[4][10]
    cdef int64_t[::1] s4_modes = stages[4][11]
    cdef double[::1] s4_amplitudes = stages[4][12]
    cdef double[::1] s4_t0 = stages[4][13]
    cdef double[::1] s4_tw = stages[4][14]
    cdef double[::1] s4_frequencies = stages[4][15]
    cdef double[:, ::1] s5_hz = stages[5][0]
    cdef double[:, ::1] s5_bz = stages[5][1]
    cdef double[:, ::1] s5_mr_hz = stages[5][2]
    cdef unsigned char[:, ::1] s5_pmc_hz = stages[5][3]
    cdef double[:, ::1] s6_hz = stages[6][0]
    cdef double[:, ::1] s6_d_hz_y = stages[6][1]
    cdef double[:, ::1] s6_d_hz_x = stages[6][2]
    cdef double s6_dx = stages[6][3]
    cdef double s6_dy = stages[6][4]
    cdef bint s6_periodic_x = stages[6][5]
    cdef bint s6_periodic_y = stages[6][6]
    cdef double[:, ::1] s7_first = stages[7][0]
    cdef double[:, ::1] s7_second = stages[7][1]
    cdef int64_t[::1] s7_targets = stages[7][2]
    cdef int64_t[::1] s7_ii = stages[7][3]
    cdef int64_t[::1] s7_jj = stages[7][4]
    cdef int64_t[::1] s7_source_ids = stages[7][5]
    cdef double[::1] s7_factors = stages[7][6]
    cdef double[::1] s7_shifts = stages[7][7]
    cdef Py_ssize_t s7_count = stages[7][8]
    cdef int64_t[::1] s7_clock = stages[7][9]
    cdef double s7_dt = stages[7][10]
    cdef int64_t[::1] s7_modes = stages[7][11]
    cdef double[::1] s7_amplitudes = stages[7][12]
    cdef double[::1] s7_t0 = stages[7][13]
    cdef double[::1] s7_tw = stages[7][14]
    cdef double[::1] s7_frequencies = stages[7][15]
    cdef double[:, ::1] s8_first = stages[8][0]
    cdef double[:, ::1] s8_second = stages[8][1]
    cdef int64_t[::1] s8_targets = stages[8][2]
    cdef int64_t[::1] s8_ii = stages[8][3]
    cdef int64_t[::1] s8_jj = stages[8][4]
    cdef double[:, ::1] s8_values = stages[8][5]
    cdef Py_ssize_t s8_count = stages[8][6]
    cdef int64_t[::1] s8_clock = stages[8][7]
    cdef double[:, ::1] s9_ex = stages[9][0]
    cdef double[:, ::1] s9_ey = stages[9][1]
    cdef double[:, ::1] s9_dx_field = stages[9][2]
    cdef double[:, ::1] s9_dy_field = stages[9][3]
    cdef double[:, ::1] s9_d_hz_y = stages[9][4]
    cdef double[:, ::1] s9_d_hz_x = stages[9][5]
    cdef double[:, ::1] s9_psi_dx_y = stages[9][6]
    cdef double[:, ::1] s9_psi_dy_x = stages[9][7]
    cdef double[:, ::1] s9_b_dx_y = stages[9][8]
    cdef double[:, ::1] s9_c_dx_y = stages[9][9]
    cdef double[:, ::1] s9_b_dy_x = stages[9][10]
    cdef double[:, ::1] s9_c_dy_x = stages[9][11]
    cdef double[:, ::1] s9_kappa_y_ex = stages[9][12]
    cdef double[:, ::1] s9_kappa_x_ey = stages[9][13]
    cdef double[:, ::1] s9_ca_ex = stages[9][14]
    cdef double[:, ::1] s9_cb_ex = stages[9][15]
    cdef double[:, ::1] s9_ca_ey = stages[9][16]
    cdef double[:, ::1] s9_cb_ey = stages[9][17]
    cdef double[:, ::1] s9_er_ex = stages[9][18]
    cdef double[:, ::1] s9_er_ey = stages[9][19]
    cdef unsigned char[:, ::1] s9_pec_ex = stages[9][20]
    cdef unsigned char[:, ::1] s9_pec_ey = stages[9][21]
    cdef double[:, :, ::1] s9_qx = stages[9][22]
    cdef double[:, :, ::1] s9_qy = stages[9][23]
    cdef double s9_curl_scale = stages[9][24]
    cdef bint s9_dispersive = stages[9][25]
    cdef double[:, ::1] s10_ez = stages[10][0]
    cdef double[:, ::1] s10_dz = stages[10][1]
    cdef double[:, ::1] s10_er_ez = stages[10][2]
    cdef unsigned char[:, ::1] s10_pec_ez = stages[10][3]
    cdef double[:, ::1] s10_sigma = stages[10][4]
    cdef double[:, ::1] s10_denominator = stages[10][5]
    cdef double[:, ::1] s10_coefficients = stages[10][6]
    cdef double[:, :, ::1] s10_r = stages[10][7]
    cdef double[:, :, ::1] s10_q = stages[10][8]
    cdef double[:, :, ::1] s10_v = stages[10][9]
    cdef double s10_dt = stages[10][10]
    cdef bint s10_dispersive = stages[10][11]
    cdef double[:, ::1] s11_ez = stages[11][0]
    cdef double[:, ::1] s11_dz = stages[11][1]
    cdef double[:, ::1] s11_er_ez = stages[11][2]
    cdef unsigned char[:, ::1] s11_pec_ez = stages[11][3]
    cdef double[:, ::1] s11_sigma = stages[11][4]
    cdef double[:, ::1] s11_denominator = stages[11][5]
    cdef double[:, ::1] s11_coefficients = stages[11][6]
    cdef double[:, :, ::1] s11_r = stages[11][7]
    cdef double[:, :, ::1] s11_q = stages[11][8]
    cdef double[:, :, ::1] s11_v = stages[11][9]
    cdef double s11_dt = stages[11][10]
    cdef bint s11_dispersive = stages[11][11]
    cdef double[:, ::1] s12_ex = stages[12][0]
    cdef double[:, ::1] s12_ey = stages[12][1]
    cdef double[:, ::1] s12_hz = stages[12][2]
    cdef double[:, ::1] s12_hz_previous = stages[12][3]
    cdef int64_t[::1] s12_point_x = stages[12][4]
    cdef int64_t[::1] s12_point_y = stages[12][5]
    cdef int64_t[::1] s12_point_offset = stages[12][6]
    cdef int64_t[::1] s12_point_stride = stages[12][7]
    cdef int64_t[::1] s12_point_it0 = stages[12][8]
    cdef int64_t[::1] s12_point_it1 = stages[12][9]
    cdef double[:, ::1] s12_monitor_values = stages[12][10]
    cdef Py_ssize_t s12_point_count = stages[12][11]
    cdef int64_t[::1] s12_clock = stages[12][12]
    cdef double[:, ::1] s13_ex = stages[13][0]
    cdef double[:, ::1] s13_ey = stages[13][1]
    cdef double[:, ::1] s13_hz = stages[13][2]
    cdef double[:, :, ::1] s13_ex_history = stages[13][3]
    cdef double[:, :, ::1] s13_ey_history = stages[13][4]
    cdef double[:, :, ::1] s13_hz_history = stages[13][5]
    cdef int64_t[::1] s13_clock = stages[13][6]
    cdef Py_ssize_t s13_stride = stages[13][7]
    cdef int64_t[::1] s14_clock = stages[14][0]
    cdef bint has_cut = len(cutcell) != 0
    cdef double[:, ::1] cut_area = cutcell[0] if has_cut else s3_mr_hz
    cdef double[:, ::1] cut_open_ex = cutcell[1] if has_cut else s3_mr_hz
    cdef double[:, ::1] cut_open_ey = cutcell[2] if has_cut else s3_mr_hz
    cdef int64_t[::1] cut_offsets = cutcell[3] if has_cut else s1_ii
    cdef int64_t[::1] cut_ii = cutcell[4] if has_cut else s1_ii
    cdef int64_t[::1] cut_jj = cutcell[5] if has_cut else s1_jj
    cdef double[::1] cut_weights = cutcell[6] if has_cut else s1_factors
    cdef NativeProgress reporter
    with nogil:
        if progress:
            progress_start(&reporter, steps)
        for step in range(steps):
            if has_cut:
                _cut_te_curl_e(s0_ex, s0_ey, s0_d_ex_y, s0_d_ey_x,
                               cut_area, cut_open_ex, cut_open_ey, s0_dx, s0_dy, nx, ny)
            else:
                _te_curl_e(s0_ex, s0_ey, s0_d_ex_y, s0_d_ey_x, s0_dx, s0_dy, nx, ny)
            _inject_events(
                s1_first, s1_second, s1_targets, s1_ii, s1_jj, s1_source_ids, s1_factors, s1_shifts,
                s1_count, s1_clock, s1_dt, s1_modes, s1_amplitudes, s1_t0, s1_tw, s1_frequencies)
            _inject_table_events(s2_first, s2_second, s2_targets, s2_ii, s2_jj, s2_values, s2_count, s2_clock)
            _te_update_h(
                s3_hz, s3_bz, s3_hz_previous, s3_d_ex_y, s3_d_ey_x, s3_psi_bz_x, s3_psi_bz_y, s3_b_bz_x,
                s3_c_bz_x, s3_b_bz_y, s3_c_bz_y, s3_kappa_x, s3_kappa_y, s3_ca_hz, s3_cb_hz, s3_mr_hz, nx,
                ny)
            if has_cut:
                _cut_merge_h(s3_hz, s3_bz, s3_mr_hz, cut_offsets,
                             cut_ii, cut_jj, cut_weights)
                _cut_zero_h(s3_hz, s3_bz, cut_area)
            _inject_events(
                s4_first, s4_second, s4_targets, s4_ii, s4_jj, s4_source_ids, s4_factors, s4_shifts,
                s4_count, s4_clock, s4_dt, s4_modes, s4_amplitudes, s4_t0, s4_tw, s4_frequencies)
            _te_finalize_h(s5_hz, s5_bz, s5_mr_hz, s5_pmc_hz, nx, ny)
            _te_curl_h(s6_hz, s6_d_hz_y, s6_d_hz_x, s6_dx, s6_dy, s6_periodic_x, s6_periodic_y, nx, ny)
            _inject_events(
                s7_first, s7_second, s7_targets, s7_ii, s7_jj, s7_source_ids, s7_factors, s7_shifts,
                s7_count, s7_clock, s7_dt, s7_modes, s7_amplitudes, s7_t0, s7_tw, s7_frequencies)
            _inject_table_events(s8_first, s8_second, s8_targets, s8_ii, s8_jj, s8_values, s8_count, s8_clock)
            _te_update_e(
                s9_ex, s9_ey, s9_dx_field, s9_dy_field, s9_d_hz_y, s9_d_hz_x, s9_psi_dx_y, s9_psi_dy_x,
                s9_b_dx_y, s9_c_dx_y, s9_b_dy_x, s9_c_dy_x, s9_kappa_y_ex, s9_kappa_x_ey, s9_ca_ex,
                s9_cb_ex, s9_ca_ey, s9_cb_ey, s9_er_ex, s9_er_ey, s9_pec_ex, s9_pec_ey, s9_qx, s9_qy,
                s9_curl_scale, s9_dispersive, nx, ny)
            if s9_dispersive:
                _finalize_e(
                    s10_ez, s10_dz, s10_er_ez, s10_pec_ez, s10_sigma, s10_denominator,
                    s10_coefficients, s10_r, s10_q, s10_v, s10_dt, s10_dispersive, nx, ny)
                _finalize_e(
                    s11_ez, s11_dz, s11_er_ez, s11_pec_ez, s11_sigma, s11_denominator,
                    s11_coefficients, s11_r, s11_q, s11_v, s11_dt, s11_dispersive, nx, ny)
            _te_sample_monitors(
                s12_ex, s12_ey, s12_hz, s12_hz_previous, s12_point_x, s12_point_y, s12_point_offset,
                s12_point_stride, s12_point_it0, s12_point_it1, s12_monitor_values, s12_point_count,
                s12_clock)
            _te_record_history(
                s13_ex, s13_ey, s13_hz, s13_ex_history, s13_ey_history, s13_hz_history, s13_clock,
                s13_stride, nx, ny)
            _advance_clock(s14_clock)
            if progress and (step + 1) % 64 == 0 and step + 1 < steps:
                progress_update(&reporter, step + 1, steps)
        if progress:
            progress_update(&reporter, steps, steps, True)

include "cuda_graph_2d.pxi"
