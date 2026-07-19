"""Device-resident Numba-CUDA runtime for the full-vector 3D solver.

All six Yee fields, twelve CPML convolution arrays, material update
coefficients, source samples, and packed plane-monitor output remain on the
GPU for the complete time loop.  Mutable state and requested monitor data are
copied to the host once, after the final step.
"""

from __future__ import annotations

import numpy as np
from numba import cuda


THREADS_1D = 128
THREADS_3D = (8, 8, 4)


@cuda.jit
def _update_h(ex, ey, ez, hx, hy, hz,
              psi_hx_y, psi_hx_z, psi_hy_x, psi_hy_z,
              psi_hz_x, psi_hz_y,
              ca_hx, cb_hx, ca_hy, cb_hy, ca_hz, cb_hz,
              x_cell_k, x_cell_b, x_cell_c,
              y_cell_k, y_cell_b, y_cell_c,
              z_cell_k, z_cell_b, z_cell_c, dx, dy, dz):
    i, j, k = cuda.grid(3)

    if i < hx.shape[0] and j < hx.shape[1] and k < hx.shape[2]:
        first = (ez[i, j + 1, k] - ez[i, j, k]) / dy
        second = (ey[i, j, k + 1] - ey[i, j, k]) / dz
        psi_y = y_cell_b[j] * psi_hx_y[i, j, k] + y_cell_c[j] * first
        psi_z = z_cell_b[k] * psi_hx_z[i, j, k] + z_cell_c[k] * second
        psi_hx_y[i, j, k] = psi_y
        psi_hx_z[i, j, k] = psi_z
        curl = (first / y_cell_k[j] + psi_y
                - second / z_cell_k[k] - psi_z)
        hx[i, j, k] = ca_hx[i, j, k] * hx[i, j, k] - cb_hx[i, j, k] * curl

    if i < hy.shape[0] and j < hy.shape[1] and k < hy.shape[2]:
        first = (ex[i, j, k + 1] - ex[i, j, k]) / dz
        second = (ez[i + 1, j, k] - ez[i, j, k]) / dx
        psi_z = z_cell_b[k] * psi_hy_z[i, j, k] + z_cell_c[k] * first
        psi_x = x_cell_b[i] * psi_hy_x[i, j, k] + x_cell_c[i] * second
        psi_hy_z[i, j, k] = psi_z
        psi_hy_x[i, j, k] = psi_x
        curl = (first / z_cell_k[k] + psi_z
                - second / x_cell_k[i] - psi_x)
        hy[i, j, k] = ca_hy[i, j, k] * hy[i, j, k] - cb_hy[i, j, k] * curl

    if i < hz.shape[0] and j < hz.shape[1] and k < hz.shape[2]:
        first = (ey[i + 1, j, k] - ey[i, j, k]) / dx
        second = (ex[i, j + 1, k] - ex[i, j, k]) / dy
        psi_x = x_cell_b[i] * psi_hz_x[i, j, k] + x_cell_c[i] * first
        psi_y = y_cell_b[j] * psi_hz_y[i, j, k] + y_cell_c[j] * second
        psi_hz_x[i, j, k] = psi_x
        psi_hz_y[i, j, k] = psi_y
        curl = (first / x_cell_k[i] + psi_x
                - second / y_cell_k[j] - psi_y)
        hz[i, j, k] = ca_hz[i, j, k] * hz[i, j, k] - cb_hz[i, j, k] * curl


@cuda.jit
def _update_e(ex, ey, ez, hx, hy, hz,
              psi_ex_y, psi_ex_z, psi_ey_x, psi_ey_z,
              psi_ez_x, psi_ez_y,
              ca_ex, cb_ex, ca_ey, cb_ey, ca_ez, cb_ez,
              x_node_k, x_node_b, x_node_c,
              y_node_k, y_node_b, y_node_c,
              z_node_k, z_node_b, z_node_c, dx, dy, dz):
    i, j, k = cuda.grid(3)

    if (i < ex.shape[0] and 0 < j < ex.shape[1] - 1
            and 0 < k < ex.shape[2] - 1):
        first = (hz[i, j, k] - hz[i, j - 1, k]) / dy
        second = (hy[i, j, k] - hy[i, j, k - 1]) / dz
        psi_y = y_node_b[j] * psi_ex_y[i, j, k] + y_node_c[j] * first
        psi_z = z_node_b[k] * psi_ex_z[i, j, k] + z_node_c[k] * second
        psi_ex_y[i, j, k] = psi_y
        psi_ex_z[i, j, k] = psi_z
        curl = (first / y_node_k[j] + psi_y
                - second / z_node_k[k] - psi_z)
        ex[i, j, k] = ca_ex[i, j, k] * ex[i, j, k] + cb_ex[i, j, k] * curl

    if (0 < i < ey.shape[0] - 1 and j < ey.shape[1]
            and 0 < k < ey.shape[2] - 1):
        first = (hx[i, j, k] - hx[i, j, k - 1]) / dz
        second = (hz[i, j, k] - hz[i - 1, j, k]) / dx
        psi_z = z_node_b[k] * psi_ey_z[i, j, k] + z_node_c[k] * first
        psi_x = x_node_b[i] * psi_ey_x[i, j, k] + x_node_c[i] * second
        psi_ey_z[i, j, k] = psi_z
        psi_ey_x[i, j, k] = psi_x
        curl = (first / z_node_k[k] + psi_z
                - second / x_node_k[i] - psi_x)
        ey[i, j, k] = ca_ey[i, j, k] * ey[i, j, k] + cb_ey[i, j, k] * curl

    if (0 < i < ez.shape[0] - 1 and 0 < j < ez.shape[1] - 1
            and k < ez.shape[2]):
        first = (hy[i, j, k] - hy[i - 1, j, k]) / dx
        second = (hx[i, j, k] - hx[i, j - 1, k]) / dy
        psi_x = x_node_b[i] * psi_ez_x[i, j, k] + x_node_c[i] * first
        psi_y = y_node_b[j] * psi_ez_y[i, j, k] + y_node_c[j] * second
        psi_ez_x[i, j, k] = psi_x
        psi_ez_y[i, j, k] = psi_y
        curl = (first / x_node_k[i] + psi_x
                - second / y_node_k[j] - psi_y)
        ez[i, j, k] = ca_ez[i, j, k] * ez[i, j, k] + cb_ez[i, j, k] * curl


@cuda.jit
def _inject_sources(ex, ey, ez, values, coords, source_ids, polarizations,
                    point_count, step):
    point = cuda.grid(1)
    if point >= point_count:
        return
    i = coords[point, 0]
    j = coords[point, 1]
    k = coords[point, 2]
    value = values[step, source_ids[point]]
    polarization = polarizations[point]
    if polarization == 0:
        cuda.atomic.add(ex, (i, j, k), value)
    elif polarization == 1:
        cuda.atomic.add(ey, (i, j, k), value)
    else:
        cuda.atomic.add(ez, (i, j, k), value)


@cuda.jit
def _sample_monitors(ex, ey, ez, hx, hy, hz, coords, point_count,
                     history, record_index):
    point = cuda.grid(1)
    if point >= point_count:
        return
    i = coords[point, 0]
    j = coords[point, 1]
    k = coords[point, 2]
    history[record_index, point, 0] = 0.25 * (
        ex[i, j, k] + ex[i, j + 1, k]
        + ex[i, j, k + 1] + ex[i, j + 1, k + 1])
    history[record_index, point, 1] = 0.25 * (
        ey[i, j, k] + ey[i + 1, j, k]
        + ey[i, j, k + 1] + ey[i + 1, j, k + 1])
    history[record_index, point, 2] = 0.25 * (
        ez[i, j, k] + ez[i + 1, j, k]
        + ez[i, j + 1, k] + ez[i + 1, j + 1, k])
    history[record_index, point, 3] = 0.5 * (
        hx[i, j, k] + hx[i + 1, j, k])
    history[record_index, point, 4] = 0.5 * (
        hy[i, j, k] + hy[i, j + 1, k])
    history[record_index, point, 5] = 0.5 * (
        hz[i, j, k] + hz[i, j, k + 1])


def _to_device(array):
    return cuda.to_device(np.ascontiguousarray(array))


def _padded_device(array, shape, dtype):
    """Upload an array, using a one-element allocation for empty metadata."""
    if array.size:
        return _to_device(np.asarray(array, dtype=dtype))
    return _to_device(np.zeros(shape, dtype=dtype))


def _blocks_3d(sim):
    return tuple((count + 1 + threads - 1) // threads for count, threads in zip(
        (sim.Nx, sim.Ny, sim.Nz), THREADS_3D))


def _copy_back(sim, state, names):
    for name in names:
        state[name].copy_to_host(getattr(sim, name))


def _finish_run(sim, steps, source_data, record_steps, history):
    times = (sim.current_step + record_steps + 1) * sim.dt
    sim.current_step += steps
    sim.monitor_results = []
    offset = 0
    for monitor in sim.monitors:
        count = len(monitor["coords"])
        fields = history[:, offset:offset + count, :].reshape(
            (len(times),) + monitor["shape"] + (6,))
        sim.monitor_results.append({
            **monitor, "dt": sim.dt, "time": times.copy(), "fields": fields,
        })
        offset += count
    sim.last_source_time = (
        sim.current_step - steps + np.arange(steps) + 1) * sim.dt
    sim.last_source_waveforms = source_data[0]
    for monitor in sim.monitor_results:
        if monitor.get("save_path") is not None:
            sim._write_plane_monitor(monitor, monitor["save_path"])
    return sim.monitor_results


def run_gpu(sim, steps, stride, progress=True, progress_desc="3D FDTD"):
    """Advance a 3D solver without host/device transfers inside the loop."""
    source_data = sim._compile_sources(steps)
    monitor_coords, _, record_steps = sim._compile_monitors(steps, stride)

    mutable_names = (
        "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
        "Psi_Hx_y", "Psi_Hx_z", "Psi_Hy_x", "Psi_Hy_z",
        "Psi_Hz_x", "Psi_Hz_y", "Psi_Ex_y", "Psi_Ex_z",
        "Psi_Ey_x", "Psi_Ey_z", "Psi_Ez_x", "Psi_Ez_y",
    )
    coefficient_names = (
        "CaEx", "CbEx", "CaEy", "CbEy", "CaEz", "CbEz",
        "CaHx", "CbHx", "CaHy", "CbHy", "CaHz", "CbHz",
    )
    state = {name: _to_device(getattr(sim, name))
             for name in mutable_names + coefficient_names}
    for axis in "xyz":
        for location in ("node", "cell"):
            for coefficient in ("k", "b", "c"):
                name = f"{axis}_{location}_{coefficient}"
                state[name] = _to_device(sim._pml[axis][f"{location}_{coefficient}"])

    values, coords, source_ids, polarizations = source_data
    source_device = {
        "values": _padded_device(values, (max(1, steps), 1), np.float64),
        "coords": _padded_device(coords, (1, 3), np.int32),
        "ids": _padded_device(source_ids, (1,), np.int32),
        "polarizations": _padded_device(polarizations, (1,), np.int8),
    }
    monitor_device = _padded_device(monitor_coords, (1, 3), np.int32)
    history_device = cuda.device_array(
        (max(1, len(record_steps)), max(1, len(monitor_coords)), 6),
        dtype=np.float64)

    source_count = len(coords)
    monitor_count = len(monitor_coords)
    source_blocks = ((source_count + THREADS_1D - 1) // THREADS_1D
                     if source_count else 0)
    monitor_blocks = ((monitor_count + THREADS_1D - 1) // THREADS_1D
                      if monitor_count else 0)
    blocks = _blocks_3d(sim)
    sim._gpu_state = state
    sim._gpu_transfer_stats = {
        "host_to_device_during_steps": 0,
        "device_to_host_during_steps": 0,
        "source_points": source_count,
        "monitor_points": monitor_count,
        "record_count": len(record_steps),
    }

    iterator = range(steps)
    progress_bar = None
    if progress:
        try:
            from tqdm.auto import tqdm
        except ImportError as exc:
            raise ImportError("Simulation progress display requires tqdm.") from exc
        progress_bar = tqdm(iterator, desc=str(progress_desc), unit="step",
                            dynamic_ncols=True)
        iterator = progress_bar
    try:
        for step in iterator:
            _update_h[blocks, THREADS_3D](
                state["Ex"], state["Ey"], state["Ez"],
                state["Hx"], state["Hy"], state["Hz"],
                state["Psi_Hx_y"], state["Psi_Hx_z"],
                state["Psi_Hy_x"], state["Psi_Hy_z"],
                state["Psi_Hz_x"], state["Psi_Hz_y"],
                state["CaHx"], state["CbHx"], state["CaHy"],
                state["CbHy"], state["CaHz"], state["CbHz"],
                state["x_cell_k"], state["x_cell_b"], state["x_cell_c"],
                state["y_cell_k"], state["y_cell_b"], state["y_cell_c"],
                state["z_cell_k"], state["z_cell_b"], state["z_cell_c"],
                sim.dx, sim.dy, sim.dz)
            _update_e[blocks, THREADS_3D](
                state["Ex"], state["Ey"], state["Ez"],
                state["Hx"], state["Hy"], state["Hz"],
                state["Psi_Ex_y"], state["Psi_Ex_z"],
                state["Psi_Ey_x"], state["Psi_Ey_z"],
                state["Psi_Ez_x"], state["Psi_Ez_y"],
                state["CaEx"], state["CbEx"], state["CaEy"],
                state["CbEy"], state["CaEz"], state["CbEz"],
                state["x_node_k"], state["x_node_b"], state["x_node_c"],
                state["y_node_k"], state["y_node_b"], state["y_node_c"],
                state["z_node_k"], state["z_node_b"], state["z_node_c"],
                sim.dx, sim.dy, sim.dz)
            if source_blocks:
                _inject_sources[source_blocks, THREADS_1D](
                    state["Ex"], state["Ey"], state["Ez"],
                    source_device["values"], source_device["coords"],
                    source_device["ids"], source_device["polarizations"],
                    source_count, step)
            if monitor_blocks and step % stride == 0:
                _sample_monitors[monitor_blocks, THREADS_1D](
                    state["Ex"], state["Ey"], state["Ez"],
                    state["Hx"], state["Hy"], state["Hz"], monitor_device,
                    monitor_count, history_device, step // stride)
        cuda.synchronize()
    finally:
        if progress_bar is not None:
            progress_bar.close()

    _copy_back(sim, state, mutable_names)
    if monitor_count:
        history = history_device.copy_to_host()[
            :len(record_steps), :monitor_count, :]
    else:
        history = np.empty((len(record_steps), 0, 6), dtype=np.float64)
    sim._gpu_state = None
    return _finish_run(sim, steps, source_data, record_steps, history)
